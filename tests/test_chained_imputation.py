import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.chained_imputation import mice_impute


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "chained_imputation"


def _json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CASES = _json("cases.json")
DISTRIBUTIONS = _json("distribution_cases.json")


def _passive(data):
    return 2 * data[:, 1] + data[:, 2]**2


def _fit(case):
    kwargs = {key: value for key, value in case.items() if key != "id"}
    kwargs["data"] = np.asarray(kwargs["data"], dtype=float)
    kwargs["methods"] = [_passive if method == "passive" else method for method in case["methods"]]
    kwargs.setdefault("seed", 55)
    return mice_impute(**kwargs)


def _check_chain(case, reference):
    result = _fit(case)
    for name in ("completed", "chain_mean", "chain_variance"):
        np.testing.assert_allclose(
            getattr(result, name), np.asarray(reference[name], dtype=float),
            atol=1e-11, rtol=1e-12, err_msg=name,
        )


def _check_distribution(case, reference):
    result = _fit(case)
    draws = result.completed[:, result.where[:, 2], 2]
    model = case["methods"][2]
    if model == "pmm":
        indices = np.asarray(case["data"], dtype=float)[:, 2]
        assert np.isin(draws, indices[np.isfinite(indices)]).all()
    if model == "logreg":
        assert np.isin(draws, [0, 1]).all()

    means = draws.mean(axis=0)
    centered = draws - means
    mean_se = np.sqrt(
        (draws.var(axis=0, ddof=1) + np.asarray(reference["standard_deviations"])**2) / len(draws)
    )
    product = centered[:, :, None] * centered[:, None, :]
    covariance = np.cov(draws, rowvar=False)
    covariance_se = np.hypot(
        product.std(axis=0, ddof=1) / np.sqrt(len(draws)), reference["covariance_se"],
    )
    assert np.all(abs(means - reference["means"]) <= 5 * mean_se + 1e-12)
    assert np.all(abs(covariance - reference["covariance"]) <= 5 * covariance_se + 1e-12)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_mice_deterministic_r_reference(case):
    _check_chain(case, _json("expected.json")["results"][case["id"]])


@pytest.mark.parametrize("case", DISTRIBUTIONS, ids=lambda case: case["id"])
def test_mice_stochastic_r_reference(case):
    _check_distribution(case, _json("distribution_expected.json")["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
@pytest.mark.parametrize("experiment,filename,cases,check", [
    ("chain", "cases.json", CASES, _check_chain),
    ("distribution", "distribution_cases.json", DISTRIBUTIONS, _check_distribution),
])
def test_mice_live_r(experiment, filename, cases, check):
    proc = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), experiment, str(FIXTURES / filename)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    references = json.loads(proc.stdout)["results"]
    for case in cases:
        check(case, references[case["id"]])


def test_seed_replays_stochastic_chains_and_preserves_observed_cells():
    data = [[0, 3], [1, 2], [2, np.nan], [3, 5], [4, 3], [5, np.nan]]
    settings = dict(methods=(None, "norm"), seed=193, m=3, iterations=2)
    initial = mice_impute(data, **settings)
    replay = mice_impute(data, **settings)
    np.testing.assert_array_equal(initial.completed, replay.completed)
    seen = ~initial.where
    for chain in initial.completed:
        np.testing.assert_array_equal(chain[seen], np.asarray(data)[seen])
    assert np.ptp(initial.completed[:, 2, 1]) > 0


def test_selective_where_leaves_unrequested_missing_cells_unchanged():
    data = [[5, 3, 1], [6, np.nan, 1], [7, 4, np.nan], [8, np.nan, 1]]
    where = np.zeros((4, 3), dtype=bool)
    where[1, 1] = True
    result = mice_impute(
        data, where=where, methods=(None, "mean", None),
        predictor_matrix=[[0, 0, 0], [1, 0, 0], [0, 0, 0]], seed=33,
    )
    np.testing.assert_allclose(result.completed[:, 1, 1], 3.5)
    assert np.isnan(result.completed[:, 2, 2]).all()
    assert np.isnan(result.completed[:, 3, 1]).all()


def test_no_predictors_and_zero_iterations_use_shared_start():
    data = [[3.0], [np.nan], [5.0]]
    initial = [[3.0], [88.0], [5.0]]
    no_sweeps = mice_impute(data, methods=["mean"], initial=initial, iterations=0, m=3)
    np.testing.assert_array_equal(no_sweeps.completed[:, 1, 0], [88] * 3)
    assert no_sweeps.chain_mean.shape == (3, 0, 1)
    after = mice_impute(data, methods=["mean"], initial=initial, iterations=1, m=3)
    np.testing.assert_array_equal(after.completed[:, 1, 0], [4] * 3)
    assert np.isnan(after.chain_variance).all()


def test_gaussian_design_handles_collinear_and_constant_predictors():
    x = np.arange(6, dtype=float)
    y = 4 + 3 * x
    y[2] = np.nan
    result = mice_impute(
        np.column_stack((x, x, np.ones_like(x), y)), methods=[None] * 3 + ["norm"],
        seed=23, m=3, iterations=2,
    )
    np.testing.assert_allclose(result.completed[:, 2, 3], [10] * 3, atol=1e-12)


def test_incomplete_predictors_are_reported_instead_of_hiding_failed_imputations():
    data = [[0, 1], [1, 2], [np.nan, np.nan], [3, 3]]
    with pytest.raises(ValueError, match="chain 1, sweep 1, column 1.*incomplete predictor"):
        mice_impute(data, methods=[None, "norm"], seed=19)


@pytest.mark.parametrize("settings,match", [
    ({"m": 0}, "^m must"),
    ({"donors": 0}, "donors must"),
    ({"match_type": 3}, "match_type"),
    ({"visit_sequence": [0]}, "visit every"),
    ({"methods": ["unsupported", "mean"]}, "methods must select"),
    ({"predictor_matrix": [[1, 0], [0, 0]]}, "own predictor"),
    ({"where": [[0, 1]]}, "boolean array"),
])
def test_mice_validation(settings, match):
    with pytest.raises(ValueError, match=match):
        mice_impute([[0, 1], [1, np.nan], [2, 4]], **settings)
