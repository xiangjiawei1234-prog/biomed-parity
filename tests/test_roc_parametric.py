import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_parametric import smooth_parametric_roc


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_parametric"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _fit(case):
    data = INPUT["datasets"][case["dataset"]]
    options = {key: value for key, value in case.items()
        if key not in ("id", "dataset", "error", "undefined_r_auc")}
    return smooth_parametric_roc(**data, **options)


def _assert_parity(case, reference):
    if "error" in case:
        assert case["error"] in reference["error"]
        with pytest.raises(ValueError, match="requires"):
            _fit(case)
        return
    if case.get("undefined_r_auc"):
        assert reference["auc"] is None and None in reference["sensitivity"]
        with pytest.raises(ValueError, match="variation|positive density"):
            _fit(case)
        return

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        actual = _fit(case)
    assert len(captured) == len(reference["warnings"])
    if captured:
        assert "below the diagonal" in reference["warnings"][0]
        assert "below the chance diagonal" in str(captured[0].message)

    scale = 100 if actual.percent else 1
    for field in ("specificity", "sensitivity", "auc"):
        np.testing.assert_allclose(getattr(actual, field), np.asarray(reference[field], dtype=float),
            rtol=3e-14, atol=5e-15 * scale, err_msg=f"{case['id']} {field}")
    assert actual.direction == reference["direction"]
    assert actual.percent == reference["percent"]
    for group in ("control_fit", "case_fit"):
        fit, expected_fit = getattr(actual, group), reference[group]
        assert fit.distribution == expected_fit["distribution"]
        assert fit.parameter_names == tuple(expected_fit["parameter_names"])
        assert fit.n_observations == expected_fit["n_observations"]
        for field in ("estimates", "standard_errors", "covariance", "log_likelihood"):
            np.testing.assert_allclose(getattr(fit, field), expected_fit[field],
                rtol=3e-14, atol=5e-15, err_msg=f"{case['id']} {group} {field}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_r(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(process.stdout)["results"]
    for case in CASES:
        _assert_parity(case, references[case["id"]])


@pytest.mark.parametrize("options,message", [
    ({"n": 0}, "positive integer"),
    ({"case_distribution": "weibull"}, "distribution must be"),
    ({"direction": "high"}, "direction must be"),
    ({"partial_auc": [0.7, 1.2]}, "partial_auc"),
])
def test_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        smooth_parametric_roc(**INPUT["datasets"]["positive_ties"], **options)


def test_repeated_noninteger_training_values_cannot_create_a_nonzero_variance():
    with pytest.raises(ValueError, match="positive variation"):
        smooth_parametric_roc([0, 0, 0, 1, 1], [0.1, 0.1, 0.1, 0.2, 0.5])
