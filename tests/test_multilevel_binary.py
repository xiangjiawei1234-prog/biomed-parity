import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.chained_imputation import mice_impute
from biomed_parity.multilevel_binary import (
    TwoLevelBinaryOptions, _LaplaceLogistic, _prepare_binary, impute_two_level_binary,
)

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "multilevel_binary"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _inputs(case):
    y = np.asarray(case["y"], dtype=float)
    x = np.asarray(case["predictors"], dtype=float)
    types = np.asarray(case["types"])
    return y, x, types


def _check_fit(case, reference):
    y, x, types = _inputs(case)
    observed = np.isfinite(y)
    fit, _, _ = _prepare_binary(y, x, types, observed, ~observed, TwoLevelBinaryOptions())
    np.testing.assert_array_equal(fit.groups, reference["groups"])
    for name in ("coefficients", "random_modes", "conditional_variances",
                 "log_likelihood", "random_variance"):
        np.testing.assert_allclose(getattr(fit, name), reference[name], atol=7e-6, rtol=3e-6, err_msg=name)
    np.testing.assert_allclose(fit.coefficient_covariance, reference["coefficient_covariance"],
                               atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_binary_random_intercept_fit_against_lme4(case):
    _check_fit(case, EXPECTED["results"][case["id"]])


def _check_moments(draws, reference, r_size):
    means = draws.mean(axis=0)
    expected_mean = np.asarray(reference["mean"])
    mean_se = np.sqrt(draws.var(axis=0, ddof=1) / len(draws)
                      + expected_mean * (1 - expected_mean) / r_size)
    assert (abs(means - expected_mean) <= 5 * mean_se + 1e-12).all()
    centered = draws - means
    products = centered[:, :, None] * centered[:, None, :]
    covariance_se = np.hypot(products.std(axis=0, ddof=1) / np.sqrt(len(draws)),
                             reference["covariance_se"])
    assert (abs(np.cov(draws, rowvar=False) - reference["covariance"]) <=
            5 * covariance_se + 1e-12).all()


def _check_draws(case, reference, strategy):
    y, x, types = _inputs(case)
    draws = impute_two_level_binary(
        y, x, types, options=TwoLevelBinaryOptions(strategy), m=2800, seed=6349,
    )
    _check_moments(draws, reference["distributions"][strategy], case["n_draws"])
    if strategy == "marginal" and reference["historical"] is not None:
        _check_moments(draws, reference["historical"], case["historical_draws"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("strategy", ["laplace", "eb", "marginal"])
def test_clustered_binary_imputations_against_r(case, strategy):
    _check_draws(case, EXPECTED["results"][case["id"]], strategy)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_live_lme4_binary_reference():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=120,
    )
    results = json.loads(run.stdout)["results"]
    for case in CASES:
        _check_fit(case, results[case["id"]])
        for strategy in ("laplace", "eb", "marginal"):
            _check_draws(case, results[case["id"]], strategy)


def test_analytic_laplace_gradient_includes_mode_dependence():
    case = CASES[0]
    y, x, _ = _inputs(case)
    keep = ~np.isnan(y)
    _, groups = np.unique(x[keep, 0], return_inverse=True)
    successes = np.bincount(groups, weights=y[keep])
    model = _LaplaceLogistic(
        np.column_stack((np.ones(keep.sum()), x[keep, 1])), y[keep], groups, successes,
        np.bincount(groups) - successes,
    )
    start = np.array([-0.2, 0.7, 1.2])
    step = 1e-5
    target = np.array([
        (model.objective(start + delta)[0] - model.objective(start - delta)[0]) / (2 * step)
        for delta in step * np.eye(len(start))
    ])
    np.testing.assert_allclose(model.objective(start)[1], target, atol=4e-8, rtol=1e-8)


def test_two_level_binary_in_chain_and_masked_predictions():
    y, x, _ = _inputs(CASES[0])
    matrix = np.column_stack((x, y, 2 * y))
    result = mice_impute(
        matrix, methods=[None, None, "2l.bin", lambda current: 2 * current[:, 2]],
        predictor_matrix=[[0, 0, 0, 0], [0, 0, 0, 0], [-2, 1, 0, 0], [0, 0, 0, 0]],
        two_level_options={2: TwoLevelBinaryOptions("eb")}, iterations=2, m=5, seed=207,
    )
    missing = np.isnan(y)
    draws = result.completed[:, :, 2]
    np.testing.assert_array_equal(draws[:, ~missing], np.broadcast_to(y[~missing], (5, (~missing).sum())))
    np.testing.assert_array_equal(result.completed[:, :, 3], 2 * draws)
    assert np.isin(draws, [0, 1]).all()
    observed = ~missing
    observed[5] = False
    target = np.zeros(len(y), dtype=bool)
    target[[2, 5, 9]] = True
    withheld = impute_two_level_binary(y, x, [-2, 1], observed=observed, where=target, m=3, seed=49)
    assert withheld.shape == (3, 3)
