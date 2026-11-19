import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.mstate_fit import msfit
from biomed_parity.mstate_prep import msprep

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "mstate_fit"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
FIELDS = (
    "times", "cumulative_hazards", "cumulative_covariance", "coefficients",
    "coefficient_covariance", "log_likelihood",
)


def _fit(case):
    dataset = INPUT["datasets"][case["dataset"]]
    options = {key: value for key, value in case.items()
               if key not in ("id", "dataset", "adjust_wide", "predict")}
    if "time" in dataset:
        records = msprep(**dataset)
        covariates = records.covariates["x"] if case.get("adjust_wide") else None
        return msfit(records, covariates=covariates, **options)
    return msfit(**(dataset | options))


def _check(case, expected):
    actual = _fit(case)
    for name in FIELDS:
        value = getattr(actual, name)
        reference = expected[name]
        if reference is None:
            assert value is None
        else:
            if name == "coefficient_covariance":
                dimension = len(actual.coefficients)
                reference = np.asarray(reference).reshape(dimension, dimension)
            np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-12, err_msg=name)
    if expected["prediction"] is not None:
        predictions = actual.predict(**case["predict"], covariance=True)
        for name, reference in expected["prediction"].items():
            np.testing.assert_allclose(
                getattr(predictions, name), reference, rtol=1e-10, atol=1e-12, err_msg=name,
            )


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


def test_risk_sets_exclude_entry_at_the_same_event_time():
    result = msfit([0, 1, 0], [1, 3, 4], [1, 1, 0], [0, 0, 0], [[0, 1]])
    np.testing.assert_array_equal(result.times, [1, 3, 4])
    np.testing.assert_array_equal(result.cumulative_hazards[:, 0], [0.5, 1.0, 1.0])
    np.testing.assert_array_equal(result.cumulative_covariance[:, 0, 0], [0.25, 0.5, 0.5])


def test_infinite_censoring_counts_as_risk_at_finite_event_times():
    result = msfit([0, 0], [2, np.inf], [1, 0], [0, 0], [[0, 1]])
    np.testing.assert_array_equal(result.times, [2])
    np.testing.assert_array_equal(result.cumulative_hazards, [[0.5]])


def test_zero_events_preserve_edges_and_prediction_states():
    result = msfit([0, 0], [2, 3], [0, 0], [0, 0], [[0, 1], [1, 2]], n_states=4)
    np.testing.assert_array_equal(result.times, [3])
    np.testing.assert_array_equal(result.cumulative_hazards, [[0, 0]])
    np.testing.assert_array_equal(result.predict().probabilities, [np.eye(4), np.eye(4)])


def test_near_ties_are_pooled_by_default():
    stop = [1, 1 + 1e-10, 2]
    result = msfit([0, 0, 0], stop, [1, 1, 0], [0, 0, 0], [[0, 1]])
    separate = msfit([0, 0, 0], stop, [1, 1, 0], [0, 0, 0], [[0, 1]], timefix=False)
    np.testing.assert_array_equal(result.times, [1, 2])
    np.testing.assert_array_equal(separate.times, stop)
    np.testing.assert_allclose(result.cumulative_hazards[:, 0], [2/3, 2/3])


@pytest.mark.parametrize(
    "options,message",
    [
        ({"t_start": [0, 2]}, "t_start < t_stop"),
        ({"t_stop": [1, np.inf]}, "finite event time"),
        ({"t_stop": [0, 2]}, "t_start < t_stop"),
        ({"t_start": [1 - 1e-10, 0]}, "zero length"),
        ({"transition": [0, 1]}, "edge indices"),
        ({"status": [1, 2]}, "0 or 1"),
        ({"covariates": [1, 2], "vartype": "greenwood"}, "only without covariates"),
        ({"covariates": [1, np.nan]}, "finite numeric columns"),
        ({"covariates": [1, 2], "new_covariates": [[1, 2]]}, "row for each edge"),
        ({"coefficients": [0.1]}, "one finite value"),
        (
            {"covariates": [1, 2], "coefficients": [0.1], "coefficient_covariance": [[-1]]},
            "positive semidefinite",
        ),
        ({"covariates": [1, 2], "coefficient_covariance": [[1]]}, "requires explicit coefficients"),
        ({"variance": "TRUE"}, "must be boolean"),
        ({"transitions": [[0, 1], [0, 2]], "vartype": "greenwood"}, "common risk set"),
        ({"status": [0, 0], "covariates": [1, 2]}, "without events"),
    ],
)
def test_invalid_input(options, message):
    arguments = {
        "t_start": [0, 0], "t_stop": [1, 2], "status": [1, 1],
        "transition": [0, 0], "transitions": [(0, 1)],
    }
    with pytest.raises(ValueError, match=message):
        msfit(**(arguments | options))
