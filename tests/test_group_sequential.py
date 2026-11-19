import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from biomed_parity.group_sequential import (
    conditional_power,
    crossing_probabilities,
    efficacy_boundaries,
    efficacy_futility_boundaries,
    symmetric_boundaries,
)
from validation.group_sequential.verify import assert_case, verify

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "group_sequential"
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


def test_single_look_is_fixed_design():
    fixed = efficacy_boundaries([1], alpha=0.025)
    np.testing.assert_allclose(fixed.z_boundaries, [norm.isf(0.025)], rtol=0, atol=1e-14)
    np.testing.assert_allclose(fixed.null_crossing_probabilities, [0.025], rtol=0, atol=1e-14)


def test_close_looks_match_adaptive_normal_integration():
    fit = efficacy_boundaries([0.1, 0.1001, 0.5, 0.99, 0.999, 1], parameter=1)
    left, right = fit.z_boundaries[:2]
    t1, t2 = fit.information_fractions[:2]
    rho = np.sqrt(t1 / t2)
    sd = np.sqrt((t2 - t1) / t2)
    exact, _ = quad(
        lambda z: norm.pdf(z) * norm.sf((right - rho * z) / sd),
        -np.inf, left, epsabs=1e-15,
    )
    assert exact == pytest.approx(fit.incremental_alpha[1], abs=5e-15)
    assert fit.null_crossing_probabilities.sum() == pytest.approx(0.025, abs=1e-12)


def test_ldof_tail_and_inactive_early_look():
    fit = efficacy_boundaries([0.0001, 0.1, 0.4, 1], spending="ldof")
    assert fit.z_boundaries[0] == np.inf
    assert fit.incremental_alpha[0] == 0
    reference_spend = 2 * norm.sf(norm.isf(0.025 / 2) / np.sqrt(0.1))
    assert fit.incremental_alpha[1] == pytest.approx(reference_spend, rel=1e-12, abs=0)
    assert norm.sf(fit.z_boundaries[1]) == pytest.approx(reference_spend, rel=1e-12, abs=0)
    with_skipped_look = efficacy_boundaries([0.1, 0.4, 1], spending="ldof")
    np.testing.assert_allclose(fit.z_boundaries[1:], with_skipped_look.z_boundaries, rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    ("fractions", "options", "message"),
    [
        ([], {}, "nonempty"),
        (["0.5", "1"], {}, "numeric"),
        ([0.25, 0.9], {}, "increase"),
        ([0.5, 0.5, 1], {}, "increase"),
        ([0, 0.5, 1], {}, "increase"),
        ([0.5, np.nan, 1], {}, "increase"),
        ([1], {"alpha": 0.5}, "alpha"),
        ([1], {"spending": "other"}, "spending"),
        ([1], {"parameter": np.inf}, "finite"),
        ([1], {"parameter": -41}, "HSD gamma"),
    ],
)
def test_invalid_design(fractions, options, message):
    with pytest.raises(ValueError, match=message):
        efficacy_boundaries(fractions, **options)


def test_symmetric_design_spends_two_tails_and_mirrors_effect():
    fit = symmetric_boundaries([0.25, 0.6, 1], alpha=0.05, parameter=0)
    args = (fit.information_fractions, fit.lower_boundaries, fit.upper_boundaries)
    under_null = crossing_probabilities(*args)
    np.testing.assert_allclose(under_null.upper.cumsum(), fit.cumulative_alpha, rtol=0, atol=1e-12)
    np.testing.assert_allclose(under_null.lower, under_null.upper, rtol=0, atol=1e-12)
    assert under_null.continuing[-1] == pytest.approx(0.90)
    positive = crossing_probabilities(*args, effect=1.6)
    negative = crossing_probabilities(*args, effect=-1.6)
    np.testing.assert_allclose(positive.upper, negative.lower, rtol=0, atol=1e-12)


@pytest.mark.parametrize("binding", [False, True], ids=["nonbinding", "binding"])
def test_futility_spending_and_final_closure(binding):
    fit = efficacy_futility_boundaries([0.3, 0.6, 0.8, 1], beta=0.15, binding=binding)
    args = (fit.information_fractions, fit.lower_boundaries, fit.upper_boundaries)
    reference_rule = crossing_probabilities(*args, respect_lower=binding)
    alternative = crossing_probabilities(*args, effect=fit.alternative_effect)
    np.testing.assert_allclose(reference_rule.upper.cumsum(), fit.cumulative_alpha, rtol=0, atol=1e-11)
    np.testing.assert_allclose(alternative.lower.cumsum(), fit.cumulative_beta, rtol=0, atol=1e-11)
    assert alternative.upper_total == pytest.approx(0.85, abs=1e-10)
    assert alternative.continuing[-1] == pytest.approx(0, abs=1e-12)
    assert fit.lower_boundaries[-1] == fit.upper_boundaries[-1]
    if not binding:
        actual_stop = crossing_probabilities(*args)
        assert actual_stop.upper_total < reference_rule.upper_total


def test_disabled_stops_allow_arbitrarily_close_observation_times():
    observed = crossing_probabilities(
        [0.5, 0.500000001, 1], [-np.inf] * 3, [3, np.inf, 2], effect=1.25,
    )
    reference = crossing_probabilities([0.5, 1], [-np.inf] * 2, [3, 2], effect=1.25)
    assert observed.upper[1] == 0
    assert observed.continuing[0] == observed.continuing[1]
    np.testing.assert_allclose(observed.upper[[0, 2]], reference.upper)


def test_conditional_power_at_last_interim_matches_normal_increment():
    fit = conditional_power([0.2, 0.68, 1], [-3, -2, -1.8], [3, 2, 1.8], look=1, z=0.55, effect=1.4)
    mean = np.sqrt(0.68) * 0.55 + 1.4 * (1 - 0.68)
    sd = np.sqrt(1 - 0.68)
    assert fit.upper_total == pytest.approx(norm.sf((1.8 - mean) / sd), abs=1e-14)
    assert fit.lower[0] == pytest.approx(norm.cdf((-1.8 - mean) / sd), abs=1e-14)
    np.testing.assert_array_equal(fit.information_fractions, [1])


def test_invalid_boundary_inputs():
    with pytest.raises(ValueError, match="must not exceed"):
        crossing_probabilities([0.5, 1], [-1, 2], [1, 1])
    with pytest.raises(ValueError, match="nonfinal look"):
        conditional_power([0.5, 1], [-1, -1], [1, 1], look=1, z=0)
    with pytest.raises(ValueError, match="beta"):
        efficacy_futility_boundaries([0.5, 1], beta=0.8)
