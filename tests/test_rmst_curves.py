import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.rmst_curves import rmst_curve_data


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "rmst_curves"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
LINKED = json.loads((FIXTURES.parent / "rmst" / "cases.json").read_text(encoding="utf-8"))
DATASETS = {dataset["id"]: dataset for dataset in LINKED + INPUT["additional_datasets"]}
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _fit(case):
    dataset = DATASETS[case["dataset"]]
    inputs = {key: value for key, value in dataset.items() if key != "id"}
    return rmst_curve_data(**inputs, pointwise_confidence_level=case.get("pointwise_confidence_level"))


def _reference_array(values):
    raw = np.asarray(values, dtype=object)
    numeric = [np.nan if value == "NA" else float(value) for value in raw.flat]
    return np.asarray(numeric).reshape(raw.shape)


def _area(polygons):
    # Positive-width rectangles have the same height on their two edges.
    widths = polygons[:, 1, 0] - polygons[:, 0, 0]
    heights = polygons[:, 3, 1] - polygons[:, 0, 1]
    assert np.all(widths > 0)
    return widths @ heights


def _assert_parity(case, reference):
    result = _fit(case)
    assert result.summary.tau == reference["tau"]
    assert result.summary.alpha == reference["alpha"]
    assert result.pointwise_confidence_level == reference["pointwise_confidence_level"]
    for name in ("arm0", "arm1"):
        curve = getattr(result, name)
        estimate = getattr(result.summary, name)
        for field, wanted in reference[name].items():
            if field in ("rmst", "rmtl"):
                value = getattr(estimate, field).estimate
            else:
                value = getattr(curve, field)
            if wanted is None:
                assert value is None
            elif field.endswith("_polygons") and wanted == []:
                assert value.shape == (0, 4, 2)
            else:
                np.testing.assert_allclose(value, _reference_array(wanted),
                    rtol=2e-14, atol=2e-14, err_msg=f"{case['dataset']} {name} {field}")
        assert _area(curve.rmst_polygons) == pytest.approx(estimate.rmst.estimate, rel=2e-14, abs=2e-14)
        assert _area(curve.rmtl_polygons) == pytest.approx(estimate.rmtl.estimate, rel=2e-14, abs=2e-14)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["dataset"])
def test_matches_recorded_survrm2_plot(case):
    _assert_parity(case, EXPECTED[case["dataset"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_survrm2_plot():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(process.stdout)["results"]
    for case in CASES:
        _assert_parity(case, references[case["dataset"]])


def test_json_export_encodes_infinite_and_missing_estimates():
    result = _fit({"dataset": "all-fail-at-origin", "pointwise_confidence_level": 0.95})
    assert np.isinf(result.arm0.log_survival_standard_error).all()
    encoded = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert encoded["arm0"]["log_survival_standard_error"] == [None]
    assert encoded["arm0"]["confidence_band"][-1] == [0.0, None, None]
    assert encoded["arm0"]["steps"] == [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    assert encoded["arm1"]["rmst_polygons"] == []
    assert encoded["summary"]["rmst_ratio"]["estimate"] is None


@pytest.mark.parametrize("level", [0, 1, -0.5, np.nan, "95%"])
def test_invalid_pointwise_confidence_level(level):
    with pytest.raises(ValueError, match="pointwise_confidence_level"):
        rmst_curve_data([1, 2], [1, 1], [0, 1], pointwise_confidence_level=level)
