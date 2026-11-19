import json
import os
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.dream import dream_random_intercept
from validation.dream.verify import assert_case, verify

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "dream"
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


def test_positive_weights_common_multiplier_preserves_inference():
    case = {key: value for key, value in CASES[1].items() if key != "id"}
    original = dream_random_intercept(**case)
    scaled_weights = 2.5 * np.asarray(case["weights"])
    scaled = dream_random_intercept(**(case | {"weights": scaled_weights}))
    np.testing.assert_allclose(scaled.coefficients, original.coefficients)
    np.testing.assert_allclose(scaled.pvalue, original.pvalue)
    np.testing.assert_allclose(scaled.df, original.df)
    assert scaled.residual_variance == pytest.approx(2.5 * original.residual_variance)
    assert scaled.random_intercept_variance == pytest.approx(original.random_intercept_variance)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"x": [[1], [1], [1]]}, "one row per"),
        ({"x": [[1, 1]] * 4}, "rank deficient"),
        ({"group": [0, 1, 2, 3]}, "repeated observation"),
        ({"group": [0, 0, 0, 0]}, "multiple groups"),
        ({"weights": [1, 1, 1, 0]}, "positive"),
        ({"contrasts": [0, 0]}, "nonzero"),
        ({"contrasts": [1]}, "one column"),
        ({"y": [1, 2, 1, 2]}, "nonzero residual"),
        ({"x": [[1, 0], [1, 0], [1, 1], [1, 1]]}, "no variation"),
    ],
)
def test_invalid_model(overrides, error):
    inputs = {"x": [[1, 0], [1, 1], [1, 0], [1, 1]], "y": [1.2, 2.5, 1.1, 3.1], "group": [0, 0, 1, 1]}
    with pytest.raises(ValueError, match=error):
        dream_random_intercept(**(inputs | overrides))
