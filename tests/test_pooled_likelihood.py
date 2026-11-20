from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.optimize import brentq, root
from scipy.special import expit
from scipy.stats import chi2

from biomed_parity.imputation import pool_d3, pool_d4


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "pooled_likelihood"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _completed_data(case):
    index = np.arange(1, case["n"] + 1)
    base = (17 * index % 37 - 18) / 12
    predictor = (11 * index % 29 - 14) / 9
    extra = (13 * index % 31 - 15) / 11
    if case["family"] == "gaussian":
        y = 120 + 3 * base + 2.3 * predictor + 1.7 * extra + 4 * (np.sin(1.4 * index) + .7 * np.cos(.7 * index))
    else:
        y = (19 * index % 43) / 43 < expit(-.45 + .63 * base + .52 * predictor + .3 * extra)

    completed = []
    for draw in range(1, case["m"] + 1):
        b = base + case["base_scale"] * (index % 6 < 3) * ((5 * index * draw + 7 * draw) % 19 - 9) / 7
        p = predictor + case["draw_scale"] * (index % 4 < 2) * ((7 * index * draw + 17 * draw + 3 * index) % 23 - 11) / 7
        e = extra + case["draw_scale"] * (index % 5 < 2) * ((3 * index * draw + 11 * draw + index) % 17 - 8) / 6
        completed.append(np.column_stack((y, np.ones(index.size), b, p, e)))
    return completed


def _gaussian_loglik(residuals, variance):
    return -(len(residuals) * np.log(2 * np.pi * variance) + np.sum(residuals**2) / variance) / 2


def _binomial_loglik(y, linear_predictor):
    return -np.logaddexp(0, (1 - 2 * y) * linear_predictor).sum()


def _linear_predictor(design, coefficients):
    return np.einsum("ij,j->i", design, coefficients)


def _fit_numeric(completed, n_parameters, gaussian):
    beta, variances, loglik = [], [], []
    for frame in completed:
        y, x = frame[:, 0], frame[:, 1:1 + n_parameters]
        if gaussian:
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            residuals = y - _linear_predictor(x, coefficients)
            variance = np.mean(residuals**2)
            variances.append(variance)
            loglik.append(_gaussian_loglik(residuals, variance))
        else:
            def score(candidate):
                residuals = y - expit(_linear_predictor(x, candidate))
                return np.einsum("ij,i->j", x, residuals)

            fitted = root(score, np.zeros(n_parameters), tol=1e-10)
            assert fitted.success
            coefficients = fitted.x
            loglik.append(_binomial_loglik(y, _linear_predictor(x, coefficients)))
        beta.append(coefficients)

    pooled_beta = np.mean(beta, axis=0)
    exact, offset_refit = [], []
    for frame in completed:
        y, x = frame[:, 0], frame[:, 1:1 + n_parameters]
        offset = _linear_predictor(x, pooled_beta)
        if gaussian:
            residuals = y - offset
            exact.append(_gaussian_loglik(residuals, np.mean(variances)))
            centered = residuals - residuals.mean()
            offset_refit.append(_gaussian_loglik(centered, np.mean(centered**2)))
        else:
            exact.append(_binomial_loglik(y, offset))
            bound = np.max(np.abs(offset)) + 30
            intercept = brentq(lambda a: np.sum(expit(offset + a) - y), -bound, bound)
            offset_refit.append(_binomial_loglik(y, offset + intercept))
    return dict(fitted=loglik, exact=exact, offset_refit=offset_refit)


def _assert_test(test, expected, rtol=1e-9):
    for key, actual in asdict(test).items():
        np.testing.assert_allclose(actual, float(expected[key]), rtol=rtol, atol=2e-11, err_msg=key)


def _check_r_case(case, reference):
    expected = reference["results"][case["id"]]
    data = _completed_data(case)
    gaussian = case["family"] == "gaussian"
    full_count = 2 + case["added"]
    full = _fit_numeric(data, full_count, gaussian)
    reduced = _fit_numeric(data, 2, gaussian)
    stacked = [np.concatenate(data, axis=0)]
    nested = {"full_loglik": full["fitted"], "reduced_loglik": reduced["fitted"], "df": case["added"]}
    restricted_method = "exact" if reference["mice_d3_exact"] else "offset_refit"
    mice_inputs = nested | {
        "restricted_full_loglik": full[restricted_method],
        "restricted_reduced_loglik": reduced[restricted_method],
    }
    _assert_test(pool_d3(**expected["mice_d3_inputs"]), expected["mice_d3"])
    _assert_test(pool_d3(**mice_inputs), expected["mice_d3"], rtol=1e-6)

    d4_inputs = nested | {
        "stacked_full_loglik": _fit_numeric(stacked, full_count, gaussian)["fitted"][0],
        "stacked_reduced_loglik": _fit_numeric(stacked, 2, gaussian)["fitted"][0],
        "full_model_df": full_count + gaussian,
    }
    for policy, output in expected["d4"].items():
        _assert_test(pool_d4(**expected["d4_inputs"], ariv=policy), output)
        _assert_test(pool_d4(**d4_inputs, ariv=policy), output, rtol=1e-6)

    if gaussian:
        exact_inputs = nested | {
            "restricted_full_loglik": full["exact"], "restricted_reduced_loglik": reduced["exact"],
        }
        for policy, output in expected["mitml_d3"].items():
            _assert_test(pool_d3(**expected["mitml_d3_inputs"], ariv=policy), output)
            _assert_test(pool_d3(**exact_inputs, ariv=policy), output)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_r(case):
    _check_r_case(case, EXPECTED)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    reference = json.loads(process.stdout)
    for case in CASES:
        _check_r_case(case, reference)


def test_identical_likelihoods_reduce_to_complete_data_chi_square():
    full, reduced = [-12, -12], [-16, -16]
    for result in (pool_d3(full, reduced, full, reduced, 2), pool_d4(full, reduced, -24, -32, 2)):
        assert result.relative_increase == 0
        assert result.df2 == np.inf
        assert result.statistic == 4
        assert result.pvalue == pytest.approx(chi2.sf(8, 2))


def test_negative_d3_and_optional_nonnegative_variance_adjustments():
    full, reduced = [-10, -10, -10], [-11, -11, -11]
    with pytest.warns(RuntimeWarning, match="D3 statistic is negative"):
        negative_d3 = pool_d3(full, reduced, [-12, -12, -12], reduced, 1)
    assert negative_d3.statistic < 0 and negative_d3.pvalue == 1
    for method, args in (
        (pool_d3, (full, reduced, full, [-11.1, -11.1, -11.1], 1)),
        (pool_d4, (full, reduced, -30, -33.3, 1)),
    ):
        assert method(*args).relative_increase == pytest.approx(-0.4)
        clipped = method(*args, ariv="positive")
        assert clipped.relative_increase == 0
        assert clipped.statistic == pytest.approx(2.2)
        assert clipped.df2 == np.inf


def test_d4_uses_mitml_near_equality_tolerance():
    result = pool_d4([-100, -110], [-100.0000007, -110.0000007], -210, -210.000000001, 1)
    assert result.relative_increase == 0
    assert result.statistic == 0
    assert result.pvalue == 1


@pytest.mark.parametrize("method,args,kwargs,message", [
    (pool_d3, ([-10], [-11], [-10], [-11], 1), {}, "at least two"),
    (pool_d3, ([-10, -10], [-11, -11], [-10], [-11], 1), {}, "restricted log likelihoods must align"),
    (pool_d3, ([-10, -10], [-11, -11], [-10, -10], [-11, np.nan], 1), {}, "finite real"),
    (pool_d4, ([-10, -10], [-11, -11], np.inf, -22, 1), {}, "finite numbers"),
    (pool_d4, ([-10, -10], [-11, -11], -20, -22, True), {}, "positive integer"),
    (pool_d4, ([-10, -10], [-11, -11], -20, -22, 1), {"ariv": "robust"}, "full_model_df"),
])
def test_invalid_likelihood_inputs(method, args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        method(*args, **kwargs)
