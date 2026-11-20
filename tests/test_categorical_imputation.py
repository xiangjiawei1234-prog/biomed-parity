import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.categorical_imputation import categorical_probabilities
from biomed_parity.chained_imputation import (
    _design, _encode_predictors, _factor_contrasts, mice_impute,
)


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "categorical_imputation"
CASES = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _settings(case):
    data = np.asarray(case["data"], dtype=float)
    levels = {data.shape[1] - 1: np.asarray(case["levels"])}
    if case["predictor_levels"] is not None:
        levels[1] = np.asarray(case["predictor_levels"])
    return data, levels


def _check_probabilities(case, expected):
    data, levels = _settings(case)
    chosen = np.arange(data.shape[1]) < data.shape[1] - 1
    observed = np.isfinite(data[:, -1])
    contrasts = {j: _factor_contrasts(len(labels), ordered=False) for j, labels in levels.items()}
    x = _encode_predictors(data, chosen, levels, contrasts)
    classes, probabilities = categorical_probabilities(
        case["method"], _design(x, observed), data[observed, -1], observed, ~observed,
        np.ones(len(data), dtype=bool), levels[data.shape[1] - 1],
    )
    np.testing.assert_array_equal(classes, case["levels"])
    np.testing.assert_allclose(probabilities, expected["probabilities"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-15, rtol=0)
    assert np.all(probabilities >= 0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_conditional_probabilities_against_r(case):
    _check_probabilities(case, EXPECTED["results"][case["id"]])


def test_orthogonal_factor_contrasts_against_r():
    for matrix in EXPECTED["contrasts"]:
        np.testing.assert_allclose(_factor_contrasts(len(matrix), ordered=True), matrix,
                                   atol=1e-13, rtol=0)


@pytest.mark.parametrize("case", [case for case in CASES if case["draws"]], ids=lambda case: case["id"])
def test_categorical_sample_frequencies_in_the_chain(case):
    data, levels = _settings(case)
    fit = mice_impute(
        data, methods=[None] * (data.shape[1] - 1) + [case["method"]],
        category_levels=levels, m=700, iterations=1, seed=781,
    )
    target = fit.where[:, -1]
    np.testing.assert_array_equal(fit.completed[:, ~target, -1],
                                  np.broadcast_to(data[~target, -1], (700, (~target).sum())))
    draws = fit.completed[:, target, -1]
    np.testing.assert_array_equal(np.isin(draws, case["levels"]), True)
    frequencies = (draws[:, :, None] == case["levels"]).mean(axis=0)
    reference = EXPECTED["results"][case["id"]]
    expected = np.asarray(reference["probabilities"])
    variance = expected * (1 - expected)
    assert (abs(frequencies - expected) <= 5 * np.sqrt(variance / len(draws)) + 2e-5).all()
    difference_se = np.sqrt(variance * (1 / len(draws) + 1 / case["draws"]))
    assert (abs(frequencies - reference["sampled"]) <= 5 * difference_se + 2e-5).all()


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_categorical_live_r():
    captured = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    live = json.loads(captured.stdout)
    for case in CASES:
        reference = live["results"][case["id"]]
        _check_probabilities(case, reference)
        if case["draws"]:
            probability = np.asarray(reference["probabilities"])
            se = np.sqrt(probability * (1 - probability) / case["draws"])
            assert (abs(reference["sampled"] - probability) <= 5 * se + 2e-5).all()


def test_categorical_draw_is_used_as_a_factor_predictor_in_the_chain():
    data = [
        [1, 10], [1, 10], [2, 100], [2, 100], [3, 35], [3, 35],
        [np.nan, np.nan], [np.nan, np.nan],
    ]
    result = mice_impute(
        data, methods=["polr", "norm.predict"], predictor_matrix=[[0, 0], [1, 0]],
        category_levels={0: [2, 3, 1]}, m=12, iterations=2, seed=56,
    )
    imputed_factors = result.completed[:, -2:, 0].astype(int)
    assert len(np.unique(imputed_factors)) == 3
    np.testing.assert_allclose(result.completed[:, -2:, 1], np.array([0, 10, 100, 35])[imputed_factors])


def test_unseen_and_ignored_levels_cannot_enter_the_response_draw():
    data = [[0, 7], [1, 7], [2, 3], [3, np.nan]]
    fit = mice_impute(
        data, methods=[None, "polyreg"], category_levels={1: [9, 3, 7]},
        ignore=[False, False, True, True], seed=112, iterations=1,
    )
    np.testing.assert_array_equal(fit.completed[:, 3, 1], 7)
    np.testing.assert_array_equal(fit.completed[:, 2, 1], 3)


def test_categorical_label_validation():
    data = [[2], [1], [np.nan]]
    with pytest.raises(ValueError, match="declared labels"):
        mice_impute(data, methods=["polr"], category_levels={0: [2, 5]})
    with pytest.raises(ValueError, match="initial values in column 0"):
        mice_impute(data, methods=["polr"], initial=[[2], [1], [5]])
    with pytest.raises(ValueError, match="50 levels"):
        mice_impute(np.r_[np.arange(51), np.nan][:, None], methods=["polyreg"])
