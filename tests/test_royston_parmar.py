import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.stats import fisk, lognorm, weibull_min

from biomed_parity.royston_parmar import royston_parmar, royston_parmar_quantile


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "royston_parmar"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
PARAMETERS = ("gamma", "knots", "scale", "timescale")


def _assert_parity(case, reference):
    parameters = {name: case[name] for name in PARAMETERS if name in case}
    actual = royston_parmar(case["time"], **parameters)
    for field, expected in reference.items():
        if field == "quantile":
            continue
        np.testing.assert_allclose(getattr(actual, field), np.asarray(expected, dtype=float),
            rtol=2e-14, atol=3e-15, err_msg=f"{case['id']} {field}")
    if "quantile" in reference:
        tail = {name: case[name] for name in ("lower_tail", "log_p") if name in case}
        actual_quantile = royston_parmar_quantile(case["probability"], **parameters, **tail)
        np.testing.assert_allclose(actual_quantile, np.asarray(reference["quantile"], dtype=float),
            rtol=5e-13, atol=1e-14, err_msg=case["id"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_flexsurv(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_flexsurv():
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(result.stdout)["results"]
    for case in CASES:
        _assert_parity(case, references[case["id"]])


@pytest.mark.parametrize("scale,reference", [
    ("hazard", weibull_min(c=1.3, scale=np.exp(0.8 / 1.3))),
    ("odds", fisk(c=1.3, scale=np.exp(0.8 / 1.3))),
    ("normal", lognorm(s=1 / 1.3, scale=np.exp(0.8 / 1.3))),
])
def test_without_interior_knots_recovers_the_standard_family(scale, reference):
    times = np.geomspace(1e-5, 10, 15)
    gamma = [-0.8, 1.3]
    result = royston_parmar(times, gamma, scale=scale)
    np.testing.assert_allclose(result.survival, reference.sf(times), rtol=3e-14, atol=0)
    np.testing.assert_allclose(result.cdf, reference.cdf(times), rtol=3e-14, atol=0)
    np.testing.assert_allclose(result.log_density, reference.logpdf(times), rtol=3e-14, atol=1e-14)
    probabilities = [0.001, 0.5, 0.9, 0.99]
    np.testing.assert_allclose(royston_parmar_quantile(probabilities, gamma, scale=scale),
        reference.ppf(probabilities), rtol=3e-14, atol=0)


def test_royston_parmar_predictor_and_rates_are_stable_far_in_identity_tail():
    gamma = [-1.5, 0.85, 0.01, -0.008]
    knots = [-1, 0.4, 1.2, 4]
    times = np.array([1e10, 1e80, 1e200])
    options = dict(knots=knots, scale="odds", timescale="identity")
    boundary = royston_parmar(knots[-1], gamma, **options)
    slope = gamma[1] - 3 * (gamma[2] * (0.4 + 1) * (4 - 0.4) + gamma[3] * (1.2 + 1) * (4 - 1.2))
    eta_boundary = np.log(np.expm1(boundary.cumulative_hazard))
    expected = eta_boundary + slope * (times - knots[-1])
    result = royston_parmar(times, gamma, **options)
    np.testing.assert_allclose(result.cumulative_hazard, expected, rtol=3e-15, atol=0)
    np.testing.assert_allclose(result.hazard, slope, rtol=3e-15, atol=0)
    assert np.isfinite(result.log_density).all()


@pytest.mark.parametrize("scale", ["hazard", "odds", "normal"])
def test_log_survival_quantile_round_trip_beyond_cdf_precision(scale):
    log_probabilities = np.array([-40.0, -100.0, -1000.0])
    gamma = [-0.2, 2.0]
    times = royston_parmar_quantile(log_probabilities, gamma, scale=scale, lower_tail=False, log_p=True)
    result = royston_parmar(times, gamma, scale=scale)
    np.testing.assert_allclose(result.log_survival, log_probabilities, rtol=6e-15, atol=0)
    assert np.isfinite(result.log_hazard).all()


def test_broadcasts_parameter_profiles_across_a_grid():
    time = np.array([0.07, 0.4, 1.8, 9.0])
    gamma = np.array([[-0.4, 1.25, 0.015], [-0.8, 1.05, -0.018]])
    knots = np.array([[-2, 0.2, 1.7], [-1, 0.7, 2.8]])
    all_rows = royston_parmar(time, gamma[:, None, :], knots=knots[:, None, :])
    all_quantiles = royston_parmar_quantile([0.2, 0.5, 0.8], gamma[:, None, :], knots=knots[:, None, :])
    for row in range(2):
        expected = royston_parmar(time, gamma[row], knots=knots[row])
        np.testing.assert_array_equal(all_rows.density[row], expected.density)
        quantiles = royston_parmar_quantile([0.2, 0.5, 0.8], gamma[row], knots=knots[row])
        np.testing.assert_array_equal(all_quantiles[row], quantiles)


def test_boundaries_and_missing_values_follow_the_documented_conventions():
    time = [-np.inf, -1, 0, np.inf, np.nan]
    result = royston_parmar(time, [-0.4, 1.0])
    np.testing.assert_array_equal(result.survival, [1, 1, 1, 0, np.nan])
    np.testing.assert_array_equal(result.cumulative_hazard, [0, 0, 0, np.inf, np.nan])
    np.testing.assert_array_equal(result.density, [0, 0, 0, 0, np.nan])
    np.testing.assert_allclose(result.hazard, [0, 0, 0, np.exp(-0.4), np.nan], rtol=1e-14)
    quantiles = royston_parmar_quantile([0, 1, np.nan], [-0.4, 1.0])
    np.testing.assert_array_equal(quantiles, [-np.inf, np.inf, np.nan])


def test_quantile_rejects_invalid_splines_even_when_both_tails_increase():
    with pytest.raises(ValueError, match="nondecreasing spline"):
        royston_parmar_quantile(0.5, [0, 0.15, 0.18, -0.18], knots=[-3, -1, 1, 3])
    with pytest.raises(ValueError, match="positive right-tail"):
        royston_parmar_quantile(0.5, [0, -0.1])


@pytest.mark.parametrize("options,message", [
    ({"gamma": [0.4]}, "final dimension"),
    ({"gamma": [0, 0.3, 0.1], "knots": [-1, 0, 0]}, "strictly increasing"),
    ({"gamma": [np.nan, 1]}, "finite"),
    ({"gamma": [[0, 1], [0, 1], [0, 1]]}, "must broadcast"),
    ({"gamma": [0.2, 1], "scale": "weibull"}, "scale"),
    ({"gamma": [0.2, 1], "timescale": "sqrt"}, "timescale"),
])
def test_rejects_malformed_parameters(options, message):
    with pytest.raises(ValueError, match=message):
        royston_parmar([0.2, 0.5], **options)


def test_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="probability must lie"):
        royston_parmar_quantile([0.5, 1.5], [0, 1])
    with pytest.raises(ValueError, match="log probability"):
        royston_parmar_quantile([0.01], [0, 1], log_p=True)
