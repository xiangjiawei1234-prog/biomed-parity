import json
import os
from dataclasses import replace

import numpy as np
import pytest

from biomed_parity.mstate_markov import markov_test, optimal_weights_matrix, optimal_weights_multiple
from biomed_parity.mstate_prep import msprep
from validation.mstate_markov.verify import CASES, HERE, assert_case, verify

REFERENCE = {
    case["id"]: case
    for case in json.loads((HERE / "expected.json").read_text(encoding="utf-8"))["cases"]
}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_r_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_r_reference():
    report = verify()
    assert report["cases_passed"] == len(CASES)
    assert report["cox_configurations_passed"] == sum(len(case.get("adjustments", [])) for case in CASES)


@pytest.fixture
def illness_death():
    return msprep(
        [[0, 1, 5], [0, 4, 6], [0, 6, 6]],
        [[0, 1, 1], [0, 1, 1], [0, 0, 1]],
        [(0, 1), (0, 2), (1, 2)],
    )


def test_landmark_ties_and_exact_logrank(illness_death):
    result = markov_test(illness_death, 2, [2, 4, 5], multipliers=[[0, 1], [2, -1], [1, 1]])
    np.testing.assert_array_equal(result.n_events_after, [2, 2, 1])
    np.testing.assert_array_equal(result.score_trace, [[-0.5, 0.5], [-0.5, 0.5], [0, 0]])
    np.testing.assert_array_equal(result.covariance[0], [[0.25, -0.25], [-0.25, 0.25]])
    np.testing.assert_array_equal(result.z_trace, [[-1, 1], [-1, 1], [0, 0]])
    np.testing.assert_array_equal(result.chi_squared_trace, [1, 1, 0])
    np.testing.assert_array_equal(result.event_subject_ids, [1, 2])
    np.testing.assert_array_equal(result.event_times, [5, 6])
    np.testing.assert_array_equal(result.bootstrap_z_trace[1], [[-2, 2], [-2, 2], [0, 0]])
    assert result.overall_statistic == pytest.approx(2 / 3)
    assert result.overall_pvalue == pytest.approx(1 / 3)
    np.testing.assert_allclose(result.state_pvalues, [1 / 3, 1 / 3], rtol=0, atol=1e-15)


@pytest.mark.parametrize("distribution", ["poisson", "normal"])
def test_seeded_replicates(illness_death, distribution):
    options = dict(distribution=distribution, n_bootstrap=7, random_state=123)
    first = markov_test(illness_death, 2, [2], **options)
    second = markov_test(illness_death, 2, [2], **options)
    np.testing.assert_array_equal(first.bootstrap_z_trace, second.bootstrap_z_trace)
    np.testing.assert_array_equal(first.bootstrap_chi_squared_trace, second.bootstrap_chi_squared_trace)
    assert first.bootstrap_z_trace.shape == (7, 1, 2)


def test_no_bootstrap_output(illness_death):
    result = markov_test(illness_death, 2, [2], n_bootstrap=0)
    assert result.bootstrap_z_trace.shape == (0, 1, 2)
    assert result.state_pvalues is result.overall_pvalue is result.z_quantiles is None
    assert result.overall_statistic == 1


def test_optimal_weights_use_state_occupancy_at_the_landmark(illness_death):
    grid = [1, 2, 4, 5, 6]
    np.testing.assert_allclose(
        optimal_weights_matrix(illness_death, 2, grid),
        [[2 / 3, 2 / 3], [2 / 3, 2 / 3], [2 / 3, 2 / 3], [0.5, 0.5], [0, 0]],
        rtol=0, atol=1e-15,
    )
    np.testing.assert_allclose(
        optimal_weights_multiple(illness_death, 2, grid, min_time=0.5),
        [2 / 9, 4 / 9, 8 / 9, 0.25, 0], rtol=0, atol=1e-15,
    )


def test_weighted_summaries_with_no_supported_grid_points(illness_death):
    result = markov_test(
        illness_death, 2, [2, 4, 5], multipliers=[[1, 1], [2, 1]],
        state_weights=[[1, 0], [0, 0], [0, 0]], overall_weights=[0, 0, 0],
    )
    np.testing.assert_array_equal(result.state_statistics, [1, np.nan])
    np.testing.assert_array_equal(result.state_pvalues, [0.5, np.nan])
    assert np.isnan(result.overall_statistic)
    assert np.isnan(result.overall_pvalue)
    assert np.isnan(result.bootstrap_overall_statistics).all()


def test_optimal_weights_align_qualifying_states_by_index():
    records = msprep(
        [[0, 0, 5, 0, 0], [0, 0, 0, 0, 7]],
        [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
        [(0, 1), (1, 2), (3, 4)], start_state=[1, 3],
    )
    np.testing.assert_array_equal(optimal_weights_matrix(records, 1, [2]), [[0.5, 0.5]])
    result = markov_test(records, 1, [2], n_bootstrap=0, state_weights="optimal")
    np.testing.assert_array_equal(result.qualifying_states, [0, 1])
    np.testing.assert_array_equal(result.state_statistics, [np.nan, 0])


def test_optimal_weights_count_future_zero_duration_events():
    records = msprep(
        [[0, 1, 5], [0, 4, 6], [0, 6, 6], [0, 0, 3]],
        [[0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]],
        [(0, 1), (0, 2), (1, 2)], start_state=[0, 0, 0, 1], start_time=[0, 0, 0, 3],
    )
    np.testing.assert_array_equal(
        optimal_weights_matrix(records, 2, [2, 3]),
        [[np.sqrt(6) / 3, np.sqrt(6) / 3], [2 / 3, 2 / 3]],
    )


def test_full_offset_trace_with_weighted_summary(illness_death):
    subject_offset = np.array([np.log(3), 0, 0])
    records = replace(illness_death, covariates={"off": subject_offset[illness_death.subject_index]})
    options = {"n_bootstrap": 0, "state_weights": "optimal", "overall_weights": "optimal"}
    legacy = markov_test(records, 2, [2], offset="off", **options)
    result = markov_test(records, 2, [2], offset="off", cox_mode="full", **options)
    np.testing.assert_array_equal(legacy.score_trace, [[-0.5, 0.5]])
    np.testing.assert_allclose(result.score_trace, [[-0.25, 0.25]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.covariance[0], [[3 / 16, -3 / 16], [-3 / 16, 3 / 16]])
    assert result.overall_statistic == pytest.approx(1 / 3)
    assert result.cox_fit.covariates == ()
    assert result.cox_fit.log_likelihood == pytest.approx(np.log(0.75))
    assert result.cox_fit.log_likelihood == legacy.cox_fit.log_likelihood


def test_cox_covariates_require_finite_nonconstant_columns(illness_death):
    records = replace(illness_death, covariates={"constant": np.ones(len(illness_death.id))})
    with pytest.raises(ValueError, match="nonconstant"):
        markov_test(records, 2, [2], n_bootstrap=0, covariates="constant")


def test_zero_duration_intervals_removed(illness_death):
    end = illness_death.t_stop.copy()
    last = np.flatnonzero(illness_death.id == 3)
    end[last] = illness_death.t_start[last]
    with pytest.warns(UserWarning, match="removed 2 rows"):
        result = markov_test(replace(illness_death, t_stop=end), 2, [2], n_bootstrap=0)
    assert result.n_subjects == 2
    np.testing.assert_array_equal(result.z_trace, [[-1, 1]])


@pytest.mark.parametrize(
    ("transition", "grid", "options", "message"),
    [
        (3, [2], {}, "zero-based index"),
        (2, [2, 1], {}, "strictly increasing"),
        (2, [2], {"multipliers": [[1, 2, 3]]}, "one column per observed transition event"),
        (2, [2], {"multipliers": [[1, 2]], "n_bootstrap": 2}, "match the number"),
        (2, [2], {"n_bootstrap": -1}, "nonnegative integer"),
        (2, [2], {"distribution": "other"}, "distribution"),
        (2, [2], {"state_weights": [[1]]}, "qualifying_states"),
        (2, [2], {"overall_weights": [-1]}, "nonnegative"),
        (2, [2], {"state_weights": "optimal", "min_time": 3}, "first grid point"),
        (2, [2], {"covariates": ["missing"]}, "absent from records"),
        (2, [2], {"covariates": ["x", "x"]}, "distinct names"),
        (2, [2], {"offset": [0, 0, 0]}, "offset must name"),
        (2, [2], {"cox_mode": "other"}, "cox_mode"),
    ],
)
def test_invalid_test_inputs(illness_death, transition, grid, options, message):
    with pytest.raises(ValueError, match=message):
        markov_test(illness_death, transition, grid, **options)
