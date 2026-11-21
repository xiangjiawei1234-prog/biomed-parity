import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.royston_parmar import royston_parmar
from biomed_parity.royston_parmar_regression import (
    fit_royston_parmar,
    royston_parmar_log_likelihood,
)


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "royston_parmar_regression"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
FIT_CASES = [case for case in CASES if case["kind"] == "fit"]
LOG_LIKELIHOOD_CASES = [case for case in CASES if case["kind"] == "loglik"]

# Observed deviations against flexsurv 2.3.2 are recorded next to the bounds:
# coefficients are limited by different stopping points of the two optimisers,
# and R's interval-censored covariance uses a numerical Hessian.
PARAMETER_ATOL = 2e-6  # max observed 1.3e-6 (ancillary gamma1 regression)
LOG_LIKELIHOOD_ATOL = 1e-9  # max observed 1.9e-11
COVARIANCE_RTOL = 1e-2  # max observed 3.7e-3 (interval-censored fit)
PREDICTION_RTOL = 2e-4  # max observed 2.6e-5 (tail rows of an ancillary fit)
PREDICTION_ATOL = 1e-9
DISTRIBUTION_FIELDS = ("survival", "cumulative_hazard", "hazard", "density", "cdf")
SUMMARY_FIELDS = {
    "survival": "survival_summary",
    "cumulative_hazard": "cumhaz_summary",
    "hazard": "hazard_summary",
}


def _arrays(case):
    left = np.array([np.inf if value is None else value for value in case["left"]], dtype=float)
    right = np.array([np.inf if value is None else value for value in case["right"]], dtype=float)
    return left, right


def _regression_options(case):
    options = {}
    if "design" in case:
        options["design"] = np.array(case["design"], dtype=float)
    if "ancillary" in case:
        options["ancillary"] = {int(index): np.array(values, dtype=float)
                                for index, values in case["ancillary"].items()}
    return options


def _likelihood_options(case):
    options = {}
    if "entry" in case:
        options["entry"] = np.array(case["entry"], dtype=float)
    if "weights" in case:
        options["weights"] = np.array(case["weights"], dtype=float)
    return options


def _distribution_options(case):
    return dict(knots=case["knots"], scale=case.get("scale", "hazard"),
                timescale=case.get("timescale", "log"))


def _expected_parameter_names(case):
    names = [f"gamma{index}" for index in range(len(case["knots"]))]
    if "design" in case:
        names.append("x")
    names.extend(f"gamma{index}(z)" for index in sorted(case.get("ancillary", {}), key=int))
    return names


def _fitted_parameters(fit):
    return np.concatenate((fit.gamma, fit.coefficients,
        *[fit.ancillary_coefficients[index] for index in sorted(fit.ancillary_coefficients)]))


def _reference_gamma(case, reference):
    left, _ = _arrays(case)
    knots = np.asarray(case["knots"], dtype=float)
    names = reference["names"]
    gamma = np.tile(np.asarray(reference["est"][:knots.size], dtype=float), (left.size, 1))
    if "design" in case:
        gamma[:, 0] += np.asarray(case["design"], dtype=float)[:, 0] * reference["est"][names.index("x")]
    for index, values in case.get("ancillary", {}).items():
        name = next(name for name in names if name.startswith(f"gamma{index}("))
        gamma[:, int(index)] += np.asarray(values, dtype=float)[:, 0] * reference["est"][names.index(name)]
    return gamma


def _assert_predictions(case, fit, reference):
    assert len(case["predict"]) == len(reference["predictions"])
    for spec, expected in zip(case["predict"], reference["predictions"]):
        times = np.asarray(spec["time"], dtype=float)
        profiles = max(len(spec.get("design", [None])), len(spec.get("entry", [None])))
        assert np.asarray(expected["survival"]).shape == (profiles, times.size)
        for profile in range(profiles):
            options = {}
            if "design" in spec:
                options["design"] = np.asarray(spec["design"], dtype=float)[profile][None, :]
            if "ancillary" in spec:
                options["ancillary"] = {int(index): np.asarray(values, dtype=float)[profile][None, :]
                                        for index, values in spec["ancillary"].items()}
            if "entry" in spec:
                actual = fit.predict(times, entry=spec["entry"][profile], **options)
            else:
                actual = fit.predict(times, **options)
            for field in DISTRIBUTION_FIELDS:
                key = SUMMARY_FIELDS.get(field, field) if "entry" not in spec else field
                if key not in expected:
                    key = field
                np.testing.assert_allclose(getattr(actual, field),
                    np.asarray(expected[key], dtype=float)[profile],
                    rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL,
                    err_msg=f"{case['id']} predict {field} profile {profile}")


def _assert_fit(case, reference):
    left, right = _arrays(case)
    fit = fit_royston_parmar(left, right, **_distribution_options(case), **_regression_options(case), **_likelihood_options(case))
    assert reference["names"] == _expected_parameter_names(case)
    np.testing.assert_allclose(_fitted_parameters(fit), reference["est"], rtol=0, atol=PARAMETER_ATOL,
        err_msg=f"{case['id']} parameters")
    np.testing.assert_allclose(fit.log_likelihood, reference["loglik"], rtol=0, atol=LOG_LIKELIHOOD_ATOL,
        err_msg=f"{case['id']} log likelihood")
    np.testing.assert_allclose(fit.covariance, reference["vcov"], rtol=COVARIANCE_RTOL, atol=1e-10,
        err_msg=f"{case['id']} covariance")
    assert fit.n_observations == len(left)
    np.testing.assert_array_equal(fit.knots, np.asarray(case["knots"], dtype=float))
    actual = royston_parmar_log_likelihood(left, right, _reference_gamma(case, reference),
        **_distribution_options(case), **_likelihood_options(case))
    np.testing.assert_allclose(actual, reference["loglik_at_est"], rtol=0, atol=LOG_LIKELIHOOD_ATOL,
        err_msg=f"{case['id']} likelihood at the reference estimate")
    _assert_predictions(case, fit, reference)
    if "weibull" in reference:
        weibull = reference["weibull"]
        np.testing.assert_allclose(fit.gamma, weibull["gamma"], rtol=0, atol=1e-7,
            err_msg=f"{case['id']} flexsurvreg weibull gamma")
        np.testing.assert_allclose(fit.log_likelihood, weibull["loglik"], rtol=0, atol=1e-10,
            err_msg=f"{case['id']} flexsurvreg weibull likelihood")
        np.testing.assert_allclose(fit.covariance, weibull["vcov"], rtol=0, atol=1e-8,
            err_msg=f"{case['id']} flexsurvreg weibull covariance")


def _assert_log_likelihood(case, reference):
    left, right = _arrays(case)
    gamma = np.asarray(case["gamma"], dtype=float)
    actual = royston_parmar_log_likelihood(left, right, gamma, **_distribution_options(case),
        **_likelihood_options(case))
    np.testing.assert_allclose(actual, reference["log_likelihood"], rtol=0, atol=LOG_LIKELIHOOD_ATOL,
        err_msg=f"{case['id']} log likelihood")


@pytest.mark.parametrize("case", FIT_CASES, ids=lambda case: case["id"])
def test_fit_matches_recorded_flexsurv(case):
    _assert_fit(case, EXPECTED[case["id"]])


@pytest.mark.parametrize("case", LOG_LIKELIHOOD_CASES, ids=lambda case: case["id"])
def test_log_likelihood_matches_recorded_flexsurv(case):
    _assert_log_likelihood(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_flexsurv():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=120,
    )
    results = json.loads(run.stdout)["results"]
    assert sorted(results) == sorted(EXPECTED)
    for case in FIT_CASES:
        _assert_fit(case, results[case["id"]])
    for case in LOG_LIKELIHOOD_CASES:
        _assert_log_likelihood(case, results[case["id"]])


def test_entry_prediction_conditions_on_survival_to_entry():
    case = next(case for case in FIT_CASES if case["id"] == "left-truncated")
    left, right = _arrays(case)
    fit = fit_royston_parmar(left, right, **_distribution_options(case), **_regression_options(case), **_likelihood_options(case))
    times = np.array([0.2, 1.0, 2.0, 4.0])
    entry = 0.5
    conditional = fit.predict(times, entry=entry)
    unconditional = royston_parmar(times, fit.gamma, **_distribution_options(case))
    initial = royston_parmar(entry, fit.gamma, **_distribution_options(case)).survival
    expected_survival = np.where(times < entry, 1.0, unconditional.survival / initial)
    np.testing.assert_allclose(conditional.survival, expected_survival, rtol=1e-14)
    np.testing.assert_allclose(conditional.cdf, 1 - expected_survival, rtol=1e-14, atol=1e-16)
    expected_hazard = np.where(times < entry, 0.0, unconditional.hazard)
    np.testing.assert_allclose(conditional.hazard, expected_hazard, rtol=1e-14, atol=1e-300)
    expected_density = np.where(times < entry, 0.0, unconditional.density / initial)
    np.testing.assert_allclose(conditional.density, expected_density, rtol=1e-14, atol=1e-300)
    before = fit.predict([0.1], entry=entry)
    np.testing.assert_array_equal(before.survival, [1.0])
    np.testing.assert_array_equal(before.cdf, [0.0])
    np.testing.assert_array_equal(before.hazard, [0.0])
    np.testing.assert_array_equal(before.density, [0.0])


def test_supplied_initial_values_reach_the_same_optimum():
    case = next(case for case in FIT_CASES if case["id"] == "weibull")
    left, right = _arrays(case)
    options = {**_distribution_options(case), **_regression_options(case), **_likelihood_options(case)}
    reference = fit_royston_parmar(left, right, **options)
    seeded = fit_royston_parmar(left, right, **options, initial=EXPECTED[case["id"]]["est"])
    np.testing.assert_allclose(_fitted_parameters(seeded), _fitted_parameters(reference),
        rtol=0, atol=1e-7)
    np.testing.assert_allclose(seeded.log_likelihood, reference.log_likelihood, rtol=0, atol=1e-10)


@pytest.mark.parametrize("options,message", [
    ({"ancillary": {0: [[0.0]]}}, "ancillary must map"),
    ({"ancillary": {3: [[0.0]]}}, "ancillary must map"),
    ({"entry": [4.0, 0.5]}, "entry must precede"),
    ({"design": [[1.0]]}, "one finite design row"),
])
def test_rejects_malformed_inputs(options, message):
    with pytest.raises(ValueError, match=message):
        fit_royston_parmar(np.array([0.5, 1.0]), np.array([0.5, np.inf]), **options)
