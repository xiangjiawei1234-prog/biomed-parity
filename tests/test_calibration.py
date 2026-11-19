import json
import os
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.calibration import validate_logistic
from validation.calibration.verify import assert_case, verify

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "calibration"
CASES = json.loads((VALIDATION / "cases.json").read_text(encoding="utf-8"))
REFERENCE = {
    case["id"]: case
    for case in json.loads((VALIDATION / "expected.json").read_text(encoding="utf-8"))["cases"]
}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_r_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_r_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_identity_bootstrap_and_failed_replicate():
    case = CASES[0]
    n = len(case["y"])
    selected = [list(range(n)), [0] * n]
    fit = validate_logistic(case["x"], case["y"], bootstrap_indices=selected, confidence=None)
    assert fit.failed_replicates == (1,)
    assert np.all(fit.counts == 1)
    np.testing.assert_allclose(fit.corrected, fit.original, rtol=0, atol=1e-8)
    assert fit.lower is None
    assert fit.upper is None


def test_seeded_bootstrap():
    case = CASES[0]
    n = len(case["y"])
    fixed = np.random.default_rng(72).integers(0, n, (8, n))
    provided = validate_logistic(case["x"], case["y"], bootstrap_indices=fixed)
    generated = validate_logistic(case["x"], case["y"], n_resamples=8, seed=72)
    np.testing.assert_array_equal(provided.corrected, generated.corrected)


def test_reject_singular_or_separated_original_model():
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        validate_logistic([[0, 0], [0, 0], [1, 1], [1, 1]], [0, 1, 0, 1])
    with pytest.raises(np.linalg.LinAlgError):
        validate_logistic([[0], [0], [1], [1]], [0, 0, 1, 1])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x": [[1], [2]]}, "one row per y"),
        ({"y": [1, 1, 1, 1]}, "both binary"),
        ({"n_resamples": 0}, "positive integer"),
        ({"confidence": 1}, "confidence"),
        ({"emax_limits": [0.8, 0.2]}, "increasing pair"),
        ({"bootstrap_indices": [[0, 1, 2]]}, "shape"),
        ({"bootstrap_indices": [[0, 1, 2, 4]]}, "zero-based"),
        ({"bootstrap_indices": [[0, 1, 2, 3]], "seed": 11}, "omitted"),
        ({"bootstrap_indices": [[0, 0, 1, 1]]}, "no successful"),
    ],
)
def test_invalid_validation(overrides, message):
    inputs = {"x": [[0], [0], [1], [1]], "y": [0, 1, 0, 1]} | overrides
    with pytest.raises(ValueError, match=message):
        validate_logistic(**inputs)
