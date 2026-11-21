import json
from math import ceil, fsum
import os

import numpy as np
import pytest

from biomed_parity.interval_bootstrap import bootstrap_interval_regression
from validation.interval_bootstrap.verify import CASES, HERE, assert_case, verify

REFERENCE = {row["id"]: row for row in json.loads((HERE / "expected.json").read_text())["cases"]}


def _inputs():
    data = REFERENCE["unweighted_ph"]
    return data["left"], [np.inf if value is None else value for value in data["right"]], np.asarray(data["design"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_interval_bootstrap_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_interval_bootstrap_matches_live_r():
    assert verify()["cases_passed"] == len(CASES)


def test_seeded_weighted_draws_match_fixed_counts():
    left, right, moderators = _inputs()
    weights = 0.25 + np.arange(len(left)) % 5 / 6
    weights[0] = 0
    draw_count = ceil(fsum(weights))
    rows = np.flatnonzero(weights)
    rng = np.random.default_rng(452)
    draws = [np.repeat(rows, rng.multinomial(draw_count, weights[rows] / fsum(weights))) for _ in range(4)]
    seeded = bootstrap_interval_regression(left, right, moderators, weights=weights, n_resamples=4, seed=452)
    fixed = bootstrap_interval_regression(left, right, moderators, weights=weights, bootstrap_indices=draws)
    np.testing.assert_array_equal(seeded.replicates, fixed.replicates)
    np.testing.assert_array_equal(seeded.covariance, fixed.covariance)
    assert seeded.failed_replicates == fixed.failed_replicates


def test_one_success_keeps_original_fit_and_missing_inference():
    left, right, moderators = _inputs()
    draws = [np.arange(len(left)), np.zeros(len(left), dtype=int)]
    result = bootstrap_interval_regression(left, right, moderators[:, 0], bootstrap_indices=draws)
    assert result.failed_replicates == (1,)
    assert result.replicates.shape == (1, 1)
    assert result.covariance.shape == (1, 1)
    np.testing.assert_array_equal(result.replicates[0], result.fit.coefficients)
    for field in ("covariance", "standard_error", "z", "pvalue", "lower", "upper"):
        assert np.isnan(getattr(result, field)).all()


def test_draw_size_uses_stable_sum_and_excludes_zero_weight_rows():
    left, right, moderators = _inputs()
    weights = np.r_[np.repeat(0.1, 30), np.zeros(10)]
    result = bootstrap_interval_regression(left, right, moderators, weights=weights, bootstrap_indices=[[0, 0, 0]])
    assert result.failed_replicates == (0,)
    assert result.replicates.shape == (0, 2)
    with pytest.raises(ValueError, match="zero-weight"):
        bootstrap_interval_regression(left, right, moderators, weights=weights, bootstrap_indices=[[0, 0, 30]])
