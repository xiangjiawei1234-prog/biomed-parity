import json
import os

import numpy as np
import pytest

from biomed_parity.rms_logistic import fit_penalized_logistic
from validation.rms_logistic_weighted.verify import CASES, HERE, assert_case, verify

REFERENCE = {row["id"]: row for row in json.loads((HERE / "expected.json").read_text())["cases"]}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_rms_logistic_weighted_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_rms_logistic_weighted_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_frequency_weights_match_replicated_rows_and_retain_zero_weight_predictions():
    x = np.arange(6, dtype=float)[:, None]
    y = np.array([1, 0, 0, 1, 0, 1])
    weights = np.array([1, 2, 0, 1, 3, 2])
    offset = np.linspace(-0.4, 0.5, len(x))
    rows = np.repeat(np.arange(len(x)), weights)
    weighted = fit_penalized_logistic(x, y, weights=weights, offset=offset)
    repeated = fit_penalized_logistic(x[rows], y[rows], offset=offset[rows])
    for field in ("coefficients", "covariance", "null_deviance", "deviance", "likelihood_ratio"):
        np.testing.assert_allclose(getattr(weighted, field), getattr(repeated, field), atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        weighted.linear_predictor, weighted.coefficients[0] + x[:, 0] * weighted.coefficients[1] + offset,
        atol=1e-12, rtol=0,
    )


def test_normalized_weights_are_scale_invariant_with_penalty():
    x = np.arange(6, dtype=float)[:, None]
    y = [1, 0, 0, 1, 0, 1]
    w = np.array([0.0, 1, 2, 0.5, 3, 2])
    first = fit_penalized_logistic(x, y, weights=w, normwt=True, penalty=4)
    scaled = fit_penalized_logistic(x, y, weights=w * 50, normwt=True, penalty=4)
    np.testing.assert_allclose(first.coefficients, scaled.coefficients, atol=1e-12, rtol=0)
    np.testing.assert_allclose(first.covariance, scaled.covariance, atol=1e-12, rtol=0)


def test_empty_design_supports_large_constant_offset():
    y = [0, 0, 1]
    plain = fit_penalized_logistic(None, y)
    shifted = fit_penalized_logistic(np.empty((len(y), 0)), y, offset=1e12)
    np.testing.assert_array_equal(plain.probabilities, shifted.probabilities)
    np.testing.assert_array_equal(plain.covariance, shifted.covariance)
    assert shifted.coefficients[0] == plain.coefficients[0] - 1e12
    assert shifted.penalty_matrix.shape == (0, 0)
    assert shifted.deviance == shifted.null_deviance
    assert shifted.effective_df == shifted.likelihood_ratio == 0
    assert shifted.pvalue == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"weights": [1, -1, 1, 1]}, "nonnegative"),
        ({"weights": [1, 0, 1, 0]}, "both outcome classes"),
        ({"weights": [1, 1]}, "per outcome"),
        ({"normwt": 1}, "boolean"),
        ({"offset": [0, 1]}, "per outcome"),
    ],
)
def test_weight_offset_validation(options, message):
    with pytest.raises(ValueError, match=message):
        fit_penalized_logistic([[0], [1], [2], [3]], [0, 1, 0, 1], **options)
