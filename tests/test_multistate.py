import json
import os
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.multistate import probtrans
from validation.multistate.verify import assert_case, verify

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "multistate"
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


def test_simultaneous_edges_only_allow_one_jump_per_time():
    fit = probtrans([1, 1, 2], [[0, 1], [1, 2]], [[0.2, 0], [0.2, 0.5], [0.2, 0.7]])
    np.testing.assert_array_equal(fit.times, [0, 1, 2])
    np.testing.assert_allclose(fit.probabilities[1, 0], [0.8, 0.2, 0])
    np.testing.assert_allclose(fit.probabilities[2, 0], [0.8, 0.16, 0.04])
    np.testing.assert_allclose(fit.probabilities.sum(axis=2), 1)


def test_fixed_and_forward_at_same_interval_give_same_point_probability():
    case = {key: value for key, value in CASES[0].items() if key != "id"}
    forward = probtrans(**case, predt=1)
    horizon = probtrans(**case, predt=9, direction="fixedhorizon")
    np.testing.assert_allclose(forward.probabilities[-1], horizon.probabilities[1])


def test_empty_prediction_interval_returns_identity():
    hazards = [[0.1], [0.6]]
    forward = probtrans([3, 9], [[0, 1]], hazards, predt=9)
    fixed = probtrans([3, 9], [[0, 1]], hazards, predt=2, direction="fixedhorizon")
    np.testing.assert_array_equal(forward.times, [9])
    np.testing.assert_array_equal(fixed.times, [0, 2])
    np.testing.assert_array_equal(forward.probabilities, [np.eye(2)])
    np.testing.assert_array_equal(fixed.probabilities, [np.eye(2), np.eye(2)])


def test_hazard_at_start_not_applied_to_later_prediction():
    fit = probtrans([0, 5], [[0, 1]], [[0.1], [0.6]])
    np.testing.assert_allclose(fit.probabilities[:, 0, 1], [0, 0.5])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"times": [2, 1]}, "nonnegative and sorted"),
        ({"transitions": [[0, 0]]}, "self-transitions"),
        ({"transitions": [[0, 1], [0, 1]]}, "distinct"),
        ({"n_states": 1}, "include every state"),
        ({"cumulative_hazards": [[0.2], [0.1]]}, "nondecreasing"),
        ({"cumulative_hazards": [[0.2], [np.inf]]}, "finite"),
        ({"cumulative_hazards": [[0.2], [1.3]]}, "jumps exceed"),
        ({"covariance": True}, "required"),
        ({"direction": "reverse"}, "direction"),
        ({"method": "other"}, "method"),
        ({"predt": -1}, "predt"),
    ],
)
def test_invalid_inputs(overrides, message):
    inputs = {"times": [1, 2], "transitions": [[0, 1]], "cumulative_hazards": [[0.1], [0.2]]} | overrides
    with pytest.raises(ValueError, match=message):
        probtrans(**inputs)
