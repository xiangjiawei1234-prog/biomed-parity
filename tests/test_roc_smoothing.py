import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_smoothing import smooth_roc


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_smoothing"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
RTOL = 2e-14
ATOL = 3e-15


def _fit(case):
    inputs = INPUT["datasets"][case["dataset"]]
    options = {key: value for key, value in case.items()
        if key not in ("id", "dataset", "error", "python_error", "compare")}
    return smooth_roc(**inputs, **options)


def _assert_parity(case, reference):
    if "error" in case:
        assert case["error"] in reference["error"]
        message = case.get("python_error",
            "distinct interior sensitivities" if case.get("method") != "density" else "positive bandwidth")
        with pytest.raises(ValueError, match=message):
            _fit(case)
        return
    if case.get("compare") == "binormal_horizontal" and "error" in reference:
        assert "missing value where TRUE/FALSE needed" in reference["error"]
        assert _fit(case).coefficients[1] == 0
        return

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        actual = _fit(case)
    if reference["warnings"]:
        assert len(captured) == 1
        assert "below the diagonal" in reference["warnings"][0]
        assert "below the chance diagonal" in str(captured[0].message)
    else:
        assert not captured

    scale = 100 if actual.percent else 1
    for field in ("specificity", "sensitivity", "auc"):
        value = getattr(actual, field)
        expected = reference[field]
        if case.get("compare") == "binormal_horizontal" and field != "auc":
            # An effectively zero R slope may flip the predicted infinite endpoints.
            value = np.sort(value[2:-2])
            expected = np.sort(expected[2:-2])
        np.testing.assert_allclose(value, np.asarray(expected, dtype=float),
            rtol=RTOL, atol=ATOL * scale, err_msg=f"{case['id']} {field}")
    for field in ("coefficients", "bandwidth"):
        if reference[field] is None:
            assert getattr(actual, field) is None
        else:
            np.testing.assert_allclose(getattr(actual, field), reference[field],
                rtol=RTOL, atol=ATOL, err_msg=f"{case['id']} {field}")
    for field in ("direction", "method", "percent"):
        assert getattr(actual, field) == reference[field]
    size = (len(case["density_controls"]) if "density_controls" in case and "density_cases" in case
        else case.get("n", 512))
    assert len(actual.specificity) == len(actual.sensitivity) == size + 2
    np.testing.assert_array_equal(actual.specificity[[0, -1]], [0, scale])
    np.testing.assert_array_equal(actual.sensitivity[[0, -1]], [scale, 0])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_p_roc(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_p_roc():
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(result.stdout)["results"]
    for case in CASES:
        _assert_parity(case, references[case["id"]])


def test_horizontal_binormal_fit_has_deterministic_endpoints():
    inputs = INPUT["datasets"]["horizontal_fit"]
    result = smooth_roc(**inputs, n=7, direction=">")
    assert result.coefficients[1] == 0
    np.testing.assert_array_equal(result.specificity[[0, 1, -2, -1]], [0, 0, 1, 1])
    np.testing.assert_allclose(result.specificity[2:-2], 2 / 3, rtol=RTOL, atol=ATOL)
    assert np.all(np.diff(result.sensitivity) <= 0)
    assert result.auc == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "fitdistr"}, "method"),
        ({"direction": "reverse"}, "direction"),
        ({"n": 1}, "at least 2"),
        ({"n": 2.5}, "integer"),
        ({"method": "density", "n": 0}, "at least 1"),
        ({"method": "density", "bw": "SJ"}, "bw"),
        ({"method": "density", "bw": -0.1}, "positive bandwidth"),
        ({"method": "density", "kernel": "uniform"}, "kernel"),
        ({"method": "density", "adjust": 0.0}, "adjust"),
        ({"method": "density", "cut": -1.0}, "cut"),
        ({"density_controls": [1]}, "require method='density'"),
        ({"method": "density", "density_controls": []}, "nonempty"),
        ({"method": "density", "density_cases": [[1, 2]]}, "one-dimensional"),
        ({"method": "density", "density_cases": [1, -1]}, "nonnegative"),
        ({"method": "density", "density_cases": [1, np.inf]}, "finite"),
        ({"method": "density", "density_cases": [1, np.nan]}, "finite"),
        ({"method": "density", "density_cases": ["1", "2"]}, "densities"),
        ({"partial_auc": (-0.3, 0.8)}, "partial_auc"),
    ],
)
def test_invalid_smoothing_options(options, message):
    with pytest.raises(ValueError, match=message):
        smooth_roc(**INPUT["datasets"]["tied"], **options)


def test_tabulated_density_rates_are_invariant_to_classwise_rescaling():
    inputs = INPUT["datasets"]["tied"]
    densities = dict(density_controls=np.array([1.0, 2.0, 4.0]), density_cases=np.array([0.0, 3.0, 7.0]))
    baseline = smooth_roc(**inputs, method="density", **densities)
    scaled = smooth_roc(**inputs, method="density",
        density_controls=densities["density_controls"] * 3e307,
        density_cases=densities["density_cases"] * 1e-312)
    np.testing.assert_allclose(baseline.specificity, scaled.specificity, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(baseline.sensitivity, scaled.sensitivity, rtol=RTOL, atol=ATOL)
    assert scaled.auc == pytest.approx(baseline.auc, rel=RTOL, abs=ATOL)
