import json
import os

import numpy as np
import pytest

from biomed_parity.rms_penalty import trace_logistic_penalties
from validation.rms_penalty.verify import CASES, HERE, assert_case, verify

REFERENCE = {row["id"]: row for row in json.loads((HERE / "expected.json").read_text())["cases"]}
X = [[-2, 1], [-1, 0], [0, 0], [1, 1], [2, 1], [3, 0]]
Y = [0, 1, 0, 0, 1, 1]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_rms_penalty_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_rms_penalty_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_scalar_tie_uses_first_row_and_coefficients_are_optional():
    result = trace_logistic_penalties(X, Y, [5, 0, 2, 5], penalty_matrix=np.zeros((2, 2)))
    np.testing.assert_array_equal(result.penalties[:, 0], [0, 5, 2, 5])
    np.testing.assert_array_equal(result.aicc, np.repeat(result.objective, 4))
    np.testing.assert_allclose(result.adjusted_covariance, result.best_fit.covariance, atol=1e-12, rtol=0)
    assert result.best_index == 0
    assert result.best_penalty == 0
    assert result.coefficients is None


@pytest.mark.parametrize(
    ("penalties", "options", "message"),
    [
        ([-1], {}, "nonnegative"),
        ([], {"include_zero": False}, "no candidates"),
        ({"other": [1]}, {}, "term types"),
        ({"simple": [1]}, {"term_types": ["simple"]}, "one recognized type"),
        ({"simple": [2], "nonlinear": [1]}, {}, "no candidates"),
        ([1], {"criterion": "likelihood"}, "criterion"),
    ],
)
def test_invalid_penalty_grid(penalties, options, message):
    with pytest.raises(ValueError, match=message):
        trace_logistic_penalties(X, Y, penalties, **options)
