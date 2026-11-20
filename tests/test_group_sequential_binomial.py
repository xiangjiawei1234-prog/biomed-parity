from itertools import product
import json
import os

import numpy as np
import pytest
from scipy.stats import binom

from biomed_parity.group_sequential_binomial import (
    binomial_conditional_probabilities,
    binomial_crossing_probabilities,
    binomial_efficacy_boundaries,
    binomial_harm_probabilities,
    binomial_sprt,
)
from validation.group_sequential_binomial.verify import CASES, HERE, assert_case, verify

REFERENCE = {
    case["id"]: case
    for case in json.loads((HERE / "expected.json").read_text(encoding="utf-8"))["cases"]
}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_r_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_r_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_first_crossings_against_enumeration():
    sizes = [2, 4, 6]
    lower = [-1, 1, 2]
    upper = [3, 3, 6]
    rate = 0.35
    result = binomial_crossing_probabilities(sizes, lower, upper, rate)
    probability = np.zeros((3, 3))
    mean_size = 0
    for trial in product((0, 1), repeat=sizes[-1]):
        mass = rate**sum(trial) * (1 - rate)**(sizes[-1] - sum(trial))
        for i, n in enumerate(sizes):
            count = sum(trial[:n])
            if count <= lower[i]:
                probability[0, i] += mass
                break
            if count >= upper[i]:
                probability[1, i] += mass
                break
            probability[2, i] += mass
        mean_size += n * mass
    np.testing.assert_allclose(result.lower[:, 0], probability[0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.upper[:, 0], probability[1], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.continuing[:, 0], probability[2], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.expected_sample_size, [mean_size], atol=1e-14, rtol=0)


def test_first_look_stops_every_trial_and_deterministic_alternatives():
    result = binomial_crossing_probabilities([2, 4], [0, 1], [1, 4], [0.0, 0.3, 1.0])
    np.testing.assert_allclose(result.lower[0], [1, 0.49, 0], atol=1e-15, rtol=0)
    np.testing.assert_allclose(result.upper[0], [0, 0.51, 1], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(result.lower[1], 0)
    np.testing.assert_array_equal(result.upper[1], 0)
    np.testing.assert_array_equal(result.continuing, 0)
    np.testing.assert_array_equal(result.expected_sample_size, [2, 2, 2])


def test_conditional_exact_single_future_look():
    result = binomial_conditional_probabilities([10, 14], [1, 4], [6, 8], [0.3, 0.6], look=0, successes=5)
    np.testing.assert_array_equal(result.sample_sizes, [14])
    np.testing.assert_allclose(result.upper[0], binom.sf(2, 4, [0.3, 0.6]), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result.lower, 0)
    np.testing.assert_array_equal(result.expected_sample_size, [14, 14])


def test_independent_harm_precedes_futility_in_overlap():
    result = binomial_harm_probabilities(
        [4, 6], [-1, 0], [2, 5], [3, 4], 0.5, test_futility=[True, False],
    )
    assert result.harm[0, 0] == pytest.approx(5 / 16)
    assert result.futility[0, 0] == pytest.approx(6 / 16)
    assert result.futility[1, 0] == 0
    np.testing.assert_allclose(result.harm + result.futility, result.upper, rtol=0, atol=1e-15)


def test_rare_futility_probability_survives_harm_dominance():
    result = binomial_harm_probabilities([8], [-1], [0], [1], 0.999)
    assert result.futility[0, 0] == pytest.approx((1 - 0.999)**8, rel=1e-12, abs=0)


def test_integer_spending_carries_forward_and_is_always_conservative():
    sizes = [2, 5, 10, 13, 30]
    lower = binomial_efficacy_boundaries(sizes, 0.4, alpha=0.05, direction="lower")
    upper = binomial_efficacy_boundaries(sizes, 0.6, alpha=0.05, direction="upper")
    np.testing.assert_array_equal(lower.sample_sizes - lower.lower_boundaries, upper.upper_boundaries)
    np.testing.assert_allclose(upper.cumulative_error, lower.cumulative_error, rtol=0, atol=1e-15)
    assert lower.lower_boundaries[0] == -1
    assert np.all(lower.cumulative_error <= lower.cumulative_alpha)
    for i, chosen in enumerate(lower.lower_boundaries):
        candidate = lower.lower_boundaries[:i + 1].copy()
        candidate[-1] = chosen + 1
        trial = binomial_crossing_probabilities(
            lower.sample_sizes[:i + 1], candidate, lower.upper_boundaries[:i + 1], 0.4,
        )
        assert trial.lower_total[0] > lower.cumulative_alpha[i]


def test_spending_accepts_boundary_at_exact_budget():
    result = binomial_efficacy_boundaries([6], 0.5, alpha=0.5**6, direction="lower")
    np.testing.assert_array_equal(result.lower_boundaries, [0])
    np.testing.assert_allclose(result.cumulative_error, [0.5**6], rtol=1e-13, atol=0)


def test_upper_spending_uses_the_original_probability_at_extreme_null():
    result = binomial_efficacy_boundaries([1], 2.0**-55, alpha=2.0**-56)
    np.testing.assert_array_equal(result.upper_boundaries, [2])
    np.testing.assert_array_equal(result.cumulative_error, [0])


def test_sprt_retains_an_uninformative_truncation():
    result = binomial_sprt(min_sample_size=1, max_sample_size=1)
    np.testing.assert_allclose(result.operating_characteristics.continuing, [[1, 1]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result.operating_characteristics.expected_sample_size, [1, 1])


def test_sprt_uses_stable_log_ratios_for_adjacent_probabilities():
    p0 = 1e-300
    p1 = np.nextafter(p0, 1.0)
    result = binomial_sprt(p0, p1, min_sample_size=1, max_sample_size=3)
    assert result.slope == pytest.approx(p0, rel=1e-14, abs=0)
    np.testing.assert_array_equal(result.operating_characteristics.sample_sizes, [3])


@pytest.mark.parametrize(
    ("sizes", "a", "b", "probabilities", "message"),
    [
        ([3, 2], [-1, 0], [1, 2], 0.5, "strictly increasing"),
        ([3, 4], [1], [1, 2], 0.5, "one threshold"),
        ([3, 4], [1, 2], [1, 2], 0.5, "strictly below"),
        ([3, 4], [1, 2], [1.5, 2], 0.5, "integer vector"),
        ([3, 4], [0, 1], [2, 3], np.nan, "finite probabilities"),
    ],
)
def test_invalid_exact_inputs(sizes, a, b, probabilities, message):
    with pytest.raises(ValueError, match=message):
        binomial_crossing_probabilities(sizes, a, b, probabilities)
