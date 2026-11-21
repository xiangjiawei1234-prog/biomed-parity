import json
import os

import numpy as np
import pytest

from biomed_parity.rms_logistic import fit_penalized_logistic
from validation.rms_logistic.verify import CASES, HERE, assert_case, verify

REFERENCE = {row["id"]: row for row in json.loads((HERE / "expected.json").read_text())["cases"]}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_rms_logistic_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_rms_logistic_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_default_categorical_penalty_is_invariant_to_reference_level():
    labels = np.repeat([0, 1, 2], 6)
    y = [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    first = fit_penalized_logistic(labels[:, None] == [1, 2], y, penalty=4, categorical_blocks=[[0, 1]])
    last = fit_penalized_logistic(labels[:, None] == [0, 1], y, penalty=4, categorical_blocks=[[0, 1]])
    np.testing.assert_allclose(first.probabilities, last.probabilities, atol=1e-12, rtol=0)
    np.testing.assert_allclose(first.effective_df, last.effective_df, atol=1e-12, rtol=0)


def test_unpenalized_collinearity_is_rejected():
    x = np.arange(6.0)
    with pytest.raises(np.linalg.LinAlgError, match="identify"):
        fit_penalized_logistic(np.column_stack((x, 2 * x + 1)), [1, 0, 0, 1, 0, 1])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"penalty": -1}, "nonnegative"),
        ({"penalty": [1]}, "one per slope"),
        ({"penalty_matrix": [[1, 2], [0, 1]]}, "symmetric"),
        ({"penalty_matrix": [[1, 2], [2, 1]]}, "positive semidefinite"),
        ({"categorical_blocks": [[0], [0]]}, "disjoint"),
        ({"categorical_blocks": [[0, 1]]}, "treatment-coded"),
    ],
)
def test_invalid_penalty_specification(options, message):
    with pytest.raises(ValueError, match=message):
        fit_penalized_logistic([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], **options)
