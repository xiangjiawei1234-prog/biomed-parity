import json
import os

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import t

from biomed_parity.rms_prediction import datadist, predict, prediction_grid, wald_predict
from validation.rms_prediction.verify import CASES, HERE, assert_case, verify

REPORT = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
REFERENCE = {case["id"]: case for case in REPORT["cases"]}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_r_reference(case):
    assert_case(case, REFERENCE[case["id"]], REPORT)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_r_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_binary_adjustment_and_categorical_levels():
    summary = datadist(
        {"binary": [1, 1, 0], "score": [10, 30, 10]},
        factor_levels={"score": [30, 10, 20]},
    )
    assert summary.limits["binary"].adjust_to == 0
    assert summary.limits["score"].adjust_to == 10
    assert "score" not in summary.values
    np.testing.assert_array_equal(prediction_grid(summary, {"score": None})["score"], [30, 10, 20])


def test_complete_factor_grid_order_and_adjustment_row():
    summary = datadist({"age": [20, 30, 40], "group": ["a", "a", "b"]})
    grid = prediction_grid(summary, {"group": None, "age": [25, 35]})
    assert tuple(grid) == ("group", "age")
    np.testing.assert_array_equal(grid["group"], ["a", "b", "a", "b"])
    np.testing.assert_array_equal(grid["age"], [25, 25, 35, 35])
    baseline = prediction_grid(summary, {}, adjust_to={"group": "b"})
    assert baseline["age"].tolist() == [30]
    assert baseline["group"].tolist() == ["b"]


def test_constant_predictor_warns_and_has_flat_ranges():
    with pytest.warns(UserWarning, match="is constant"):
        summary = datadist({"age": [42, None, 42]})
    assert summary.limits["age"].effect == (42, 42)
    assert summary.limits["age"].display == (42, 42)
    assert summary.values["age"] == (42,)


def test_reference_contrast_uses_shared_covariance_and_t_critical_value():
    result = wald_predict(
        [[1, 2], [1, 3]], [1, 2], [[4, 0.5], [0.5, 1]],
        reference=[1, 2], residual_df=9, confidence_level=0.9,
    )
    np.testing.assert_array_equal(result.estimate, [0, 2])
    np.testing.assert_array_equal(result.standard_error, [0, 1])
    np.testing.assert_allclose(result.lower, [0, 2 - t.ppf(0.95, 9)], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.upper, [0, 2 + t.ppf(0.95, 9)], atol=1e-12, rtol=0)


def test_simultaneous_normal_limits_match_independent_family_probability():
    covariance = np.diag([1e-12, 4, 9])
    result = wald_predict(np.eye(3), [0, 0, 0], covariance, confidence_type="simultaneous")
    expected = ndtri((1 + 0.95**(1 / 3)) / 2)
    np.testing.assert_allclose(result.upper / result.standard_error, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(result.lower, -result.upper)


def test_simultaneous_limits_are_invariant_to_repeated_and_reordered_rows():
    design = np.array([[1, 0, 1], [1, 2, 0], [1, 1, 2], [1, -1, 1]])
    options = dict(coefficients=[0, 0, 0], covariance=np.eye(3), confidence_type="simultaneous")
    original = wald_predict(design, **options)
    order = [3, 0, 1, 3, 2, 0]
    repeated = wald_predict(np.vstack((design[order], -design[1])), **options)
    np.testing.assert_array_equal(repeated.upper, original.upper[[*order, 1]])


def test_simultaneous_rank_one_t_limits_preserve_zero_reference_and_endpoint_order():
    options = dict(reference=[1, 0], residual_df=9, transform=lambda x: np.exp(-x))
    args = ([[1, 0], [1, 2], [1, -1], [1, 2]], [1, 2], [[4, 0.5], [0.5, 1]])
    result = wald_predict(*args, confidence_type="simultaneous", **options)
    pointwise = wald_predict(*args, **options)
    np.testing.assert_array_equal(result.lower, pointwise.lower)
    np.testing.assert_array_equal(result.upper, pointwise.upper)
    assert result.lower[0] == result.upper[0] == 1
    assert np.all(result.lower <= result.upper)


def test_simultaneous_mode_rejects_indefinite_coefficient_covariance():
    with pytest.raises(ValueError, match="positive semidefinite covariance"):
        wald_predict(np.eye(2), [0, 0], [[1, 2], [2, 1]], confidence_type="simultaneous")


def test_transform_and_point_only_output():
    fit = wald_predict([[3]], [2], [[1]], transform=np.exp, confidence_level=0)
    assert fit.estimate[0] == np.exp(6)
    assert fit.linear_predictor[0] == 6
    assert fit.standard_error[0] == 3
    assert fit.lower is fit.upper is None


@pytest.mark.parametrize(
    ("data", "options", "message"),
    [
        ({"a": [1, 2], "b": [1, 2, 3]}, {}, "same length"),
        ({"a": [1, None]}, {}, "nonmissing"),
        ({"a": [1, 2]}, {"q_effect": [0.4, 0.25]}, "ordered probabilities"),
        ({"a": ["a", "b"]}, {"factor_levels": {"a": ["a"]}}, "outside its declared"),
    ],
)
def test_invalid_summaries(data, options, message):
    with pytest.raises(ValueError, match=message):
        datadist(data, **options)


def test_invalid_prediction_grid_and_covariance():
    summary = datadist({"x": np.arange(11), "z": np.arange(11)})
    with pytest.raises(ValueError, match="100,000"):
        prediction_grid(summary, {"x": None, "z": None}, points=400)
    with pytest.raises(ValueError, match="known predictor"):
        prediction_grid(summary, {"unknown": 1})
    with pytest.raises(ValueError, match="negative prediction variance"):
        wald_predict([[1, 1]], [1, 1], [[1, -2], [-2, 1]])
    with pytest.raises(ValueError, match="confidence_type"):
        wald_predict([[1]], [1], [[1]], confidence_type="individual")
    with pytest.raises(ValueError, match="one row per prediction setting"):
        predict(summary, {"x": [1, 2]}, lambda grid: [[1]], [2], [[1]])
