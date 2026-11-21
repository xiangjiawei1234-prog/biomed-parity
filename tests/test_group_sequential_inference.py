import json
import os

import numpy as np
import pytest
from scipy.stats import norm

from biomed_parity.group_sequential import efficacy_boundaries
from biomed_parity.group_sequential_inference import (
    repeated_confidence_intervals,
    repeated_pvalues,
    sequential_pvalue,
)
from validation.group_sequential_inference.verify import CASES, HERE, assert_case, verify

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


def test_first_look_analytical_spending_inverse():
    assert sequential_pvalue([0.5], [2], parameter=0) == pytest.approx(norm.sf(2) / 0.5, abs=1e-12)
    assert sequential_pvalue([0.5], [2], spending="ldof") == pytest.approx(
        2 * norm.sf(norm.isf(norm.sf(2) / 2) * np.sqrt(0.5)), abs=1e-12
    )
    assert sequential_pvalue([1], [10], search_interval=[1e-40, 0.99]) == pytest.approx(
        norm.sf(10), rel=1e-10, abs=0
    )


def test_inverting_efficacy_thresholds_recovers_alpha_at_each_look():
    fractions = [0.3, 0.6, 1]
    bounds = efficacy_boundaries(fractions, alpha=0.025).z_boundaries
    pvalues = repeated_pvalues(fractions, bounds)
    np.testing.assert_allclose(pvalues.repeated, 0.025, atol=1e-12, rtol=0)
    assert pvalues.sequential == pytest.approx(0.025, abs=1e-12)


def test_schedule_units_and_unobserved_future_look():
    relative = repeated_pvalues([0.33, 0.62], [2.8, 1.65])
    absolute = repeated_pvalues([33, 62], [2.8, 1.65], planned_final_information=100)
    np.testing.assert_allclose(relative.repeated, absolute.repeated, rtol=1e-14, atol=0)
    final = repeated_pvalues([0.33, 0.62, 1.0], [2.8, 1.65, 1.3])
    np.testing.assert_allclose(absolute.cumulative, final.cumulative[:2], atol=1e-12, rtol=0)


def test_intervals_handle_inactive_and_reflected_bounds():
    fit = repeated_confidence_intervals([0.3, 0.5], [0.1, 0.2], [np.inf, 2], alternative="less")
    np.testing.assert_array_equal(fit.lower, [-np.inf, -np.inf])
    np.testing.assert_array_equal(fit.upper, [np.inf, 0.9])


@pytest.mark.parametrize(
    ("information", "z", "options", "message"),
    [
        ([0.5, 0.4], [1, 2], {}, "strictly increasing"),
        ([0.5, 1], [1], {}, "same length"),
        ([0.5, 1], [1, 2], {"spending_times": [0.4, 0.4]}, "strictly increasing"),
        ([0.5, 1], [1, 2], {"spending_times": [1, 1.1]}, "at most the last"),
        ([0.5], [np.nan], {}, "finite numbers"),
        ([0.5], [1], {"planned_final_information": 0}, "positive and finite"),
        ([0.5], [1], {"search_interval": [0, 1]}, "strictly between"),
    ],
)
def test_invalid_monitoring_inputs(information, z, options, message):
    with pytest.raises(ValueError, match=message):
        sequential_pvalue(information, z, **options)
