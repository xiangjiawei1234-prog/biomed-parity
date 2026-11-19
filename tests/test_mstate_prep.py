import json
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.mstate_prep import MsprepWarning, msprep

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "mstate_prep"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
FIELDS = {
    "id": "id",
    "from": "from_state",
    "to": "to_state",
    "trans": "transition",
    "Tstart": "t_start",
    "Tstop": "t_stop",
    "time": "time",
    "status": "status",
}


def _records(case):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", MsprepWarning)
        output = msprep(**case["arguments"])
    assert all(item.category is MsprepWarning for item in captured)
    return output, [str(item.message) for item in captured]


def _check(case, expected):
    actual, messages = _records(case)
    assert len(messages) == len(expected["warnings"])
    for message, reference in zip(messages, expected["warnings"], strict=True):
        if "simultaneous transitions" in reference:
            assert "simultaneous" in message
        else:
            assert "status=0" in reference and "a censored time precedes" in message

    difference = case.get("difference")
    if difference == "r-empty-result-error":
        assert "same length" in expected["error"]
        assert actual.id.size == 0
        return

    keep = np.ones(len(actual.id), dtype=bool)
    if difference == "r-omits-singleton":
        extra = (actual.id == 1) & (actual.from_state == 1)
        assert np.flatnonzero(extra).size == 1
        assert actual.transition[extra].tolist() == [2]
        assert actual.t_start[extra].tolist() == [2]
        assert actual.t_stop[extra].tolist() == [5]
        assert actual.status[extra].tolist() == [1]
        keep = ~extra
    for key, field in FIELDS.items():
        observed = getattr(actual, field)[keep]
        reference = [float("inf") if item == "Inf" else item for item in expected["rows"][key]]
        np.testing.assert_array_equal(observed, reference, err_msg=key)
    assert actual.covariates.keys() == expected["rows"].keys() - FIELDS.keys()
    for name, column in actual.covariates.items():
        reference = [None if item == "NA" else item for item in expected["rows"][name]]
        np.testing.assert_array_equal(column[keep], reference, err_msg=name)


@pytest.mark.parametrize("case", CASES, ids=lambda entry: entry["id"])
def test_matches_recorded_r(case):
    _check(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(result.stdout)["results"]
    for case in CASES:
        _check(case, references[case["id"]])


def test_matrix_and_edges_are_interchangeable():
    case = CASES[0]
    actual, _ = _records(case)
    edges = [[0, 1], [0, 2], [1, 2]]
    np.testing.assert_array_equal(actual.transitions, edges)
    np.testing.assert_array_equal(actual.transition_matrix, [
        [np.nan, 1, 2], [np.nan, np.nan, 3], [np.nan, np.nan, np.nan],
    ])
    explicit = {**case["arguments"], "transition_matrix": None, "transitions": edges}
    converted = msprep(**explicit)
    for field in ("subject_index", *FIELDS.values()):
        np.testing.assert_array_equal(getattr(actual, field), getattr(converted, field))
    inputs = case["arguments"]["data"]["subject_code"]
    assert actual.id.tolist() == [inputs[i] for i in actual.subject_index]


def test_subject_at_absorbing_state_never_enters_risk_set():
    result = msprep(
        [[None, 1, 5], [None, 1, 6]],
        [[None, 1, 1], [None, 1, 1]],
        [(0, 1), (1, 2)],
        start_state=[0, 2],
        keep=[100, 200],
    )
    np.testing.assert_array_equal(result.subject_index, [0, 0])
    np.testing.assert_array_equal(result.status, [1, 1])
    np.testing.assert_array_equal(result.covariates["keep1"], [100, 100])


def test_graph_with_no_edges_has_valid_empty_output():
    result = msprep([[None, None]], [[None, None]], [])
    assert result.status.shape == (0,)
    assert result.transitions.shape == (0, 2)
    np.testing.assert_array_equal(result.transition_matrix, np.full((2, 2), np.nan))


@pytest.mark.parametrize(
    "options,message",
    [
        ({"transitions": [(0, 1), (1, 0)]}, "acyclic"),
        ({"transitions": [(0, 1), (0, 1)]}, "distinct"),
        ({"transitions": [(1, 1)]}, "different states"),
        ({"transitions": [(0, 3)]}, "within the state columns"),
        ({"transitions": [(0.2, 1)]}, "integer state pairs"),
        (
            {"transitions": None, "transition_matrix": [[None, 2, 1], [None, None, 4],
                                                        [None, None, None]]},
            "consecutively",
        ),
        ({"transition_matrix": [[None, 1, None], [None] * 3, [None] * 3]}, "exactly one"),
        ({"status": [[None, 0, 1]]}, "same shape"),
        ({"status": [[None, 0, 2], [None, 0, 1]]}, "0 or 1"),
        ({"time": [[None, 3, 4], [None, 2, None]]}, "all destination states"),
        ({"start_state": [0, 3]}, "within the state columns"),
        ({"start_time": [-1, 1]}, "nonnegative"),
        ({"start_time": [0, 0, 0]}, "for each subject"),
        ({"id": ["a", "a"]}, "uniquely"),
        ({"keep": {"label": ["a"]}}, "one row per subject"),
    ],
)
def test_invalid_input(options, message):
    arguments = {
        "time": [[None, 3, 4], [None, 2, 6]],
        "status": [[None, 0, 1], [None, 0, 1]],
        "transitions": [(0, 1), (0, 2), (1, 2)],
    }
    with pytest.raises(ValueError, match=message):
        msprep(**(arguments | options))
