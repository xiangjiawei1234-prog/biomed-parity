import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from biomed_parity.group_sequential import efficacy_boundaries
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
