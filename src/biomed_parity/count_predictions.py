"""Prediction and simulation for fitted zero-inflated count mixed models."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from .zero_inflated import (
    ZeroInflatedNegativeBinomialResult, ZeroInflatedPoissonResult, _group_codes, _matrix,
)


CountFit = ZeroInflatedPoissonResult | ZeroInflatedNegativeBinomialResult


@dataclass(frozen=True)
class CountPredictions:
    """One vector per prediction scale, in the supplied observation order."""

    link: NDArray[np.float64]
    conditional_mean: NDArray[np.float64]
    inflation_logit: NDArray[np.float64]
    structural_zero_probability: NDArray[np.float64]
    response_mean: NDArray[np.float64]
    zero_probability: NDArray[np.float64]
    dispersion: NDArray[np.float64]


def _rows(*values):
    sizes = []
    for value in values:
        if value is not None:
            shape = np.shape(value)
            if not shape or shape[0] == 0:
                raise ValueError("prediction inputs must have a nonempty observation dimension")
            sizes.append(shape[0])
    if len(set(sizes)) > 1:
        raise ValueError("prediction inputs must have the same number of observations")
    return sizes[0] if sizes else 1


def _linear_predictors(fit, groups, count_design, inflation_design, dispersion_design, offset):
    if not isinstance(fit, (ZeroInflatedPoissonResult, ZeroInflatedNegativeBinomialResult)):
        raise ValueError("fit must be a fitted zero-inflated random-intercept count model")
    n = _rows(groups, count_design, inflation_design, dispersion_design, offset)
    offset = np.zeros(n) if offset is None else np.asarray(offset)
    if (offset.shape != (n,) or offset.dtype.kind not in "iuf" or not np.isfinite(offset).all()):
        raise ValueError("offset must contain one finite number per observation")

    def linear(design, coefficients, name):
        matrix = _matrix(design, n, name)
        if matrix.shape[1] != len(coefficients):
            raise ValueError(f"{name} columns must match the fitted coefficients")
        return np.einsum("ij,j->i", matrix, coefficients)

    beta = linear(count_design, fit.count_coefficients, "count_design") + offset
    gamma = linear(inflation_design, fit.inflation_coefficients, "inflation_design")
    if isinstance(fit, ZeroInflatedNegativeBinomialResult):
        log_shape = linear(dispersion_design, fit.dispersion_coefficients, "dispersion_design")
        if not np.isfinite(log_shape).all() or (np.abs(log_shape) > 35).any():
            raise ValueError("log dispersion predictor exceeds the supported numerical range")
    elif dispersion_design is not None:
        raise ValueError("dispersion_design applies only to NB2 models")
    else:
        log_shape = None
    if not (np.isfinite(beta).all() and np.isfinite(gamma).all()):
        raise ValueError("linear predictors must be finite")
    return beta, gamma, log_shape


def _missing_group(label):
    return label is None or (isinstance(label, Real) and np.isnan(label))


def predict_count_mixed(
    fit: CountFit,
    *,
    groups: ArrayLike | None = None,
    count_design: ArrayLike | None = None,
    inflation_design: ArrayLike | None = None,
    dispersion_design: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    allow_new_levels: bool = False,
) -> CountPredictions:
    """Predict on count, structural-zero, unconditional and dispersion scales.

    A known group uses its fitted conditional intercept. Missing group labels
    or an omitted group vector set the random intercept to zero. An unseen
    group is accepted at zero only when allow_new_levels is true.
    """
    if not isinstance(allow_new_levels, (bool, np.bool_)):
        raise ValueError("allow_new_levels must be boolean")
    eta, gamma, log_shape = _linear_predictors(
        fit, groups, count_design, inflation_design, dispersion_design, offset,
    )
    if groups is not None:
        labels = np.asarray(groups, dtype=object)
        if labels.shape != eta.shape:
            raise ValueError("groups must contain one label per observation")
        fitted_modes = dict(zip(fit.group_labels, fit.random_intercepts, strict=True))
        for row, label in enumerate(labels):
            if _missing_group(label):
                continue
            if not isinstance(label, (Real, str, bytes, np.bool_)):
                raise ValueError("groups must contain numeric or string labels")
            if label not in fitted_modes and not allow_new_levels:
                raise ValueError(f"group {label!r} was absent from the fitted model")
            eta[row] += fitted_modes.get(label, 0.0)
    if (eta >= np.log(np.finfo(float).max)).any():
        raise ValueError("count predictor exceeds the supported numerical range")
    mean = np.exp(eta)
    zi = expit(gamma)
    count_weight = expit(-gamma)
    dispersion = np.ones_like(mean) if log_shape is None else np.exp(log_shape)
    if log_shape is None:
        ordinary_zero = np.exp(-mean)
    else:
        ordinary_zero = np.exp(-dispersion * np.logaddexp(0, eta - log_shape))
    return CountPredictions(
        eta, mean, gamma, zi, mean * count_weight, zi + count_weight * ordinary_zero, dispersion,
    )


def simulate_count_mixed(
    fit: CountFit,
    groups: ArrayLike,
    *,
    count_design: ArrayLike | None = None,
    inflation_design: ArrayLike | None = None,
    dispersion_design: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    nsim: int = 1,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """Draw new cluster intercepts and count responses at fitted parameters."""
    if not isinstance(nsim, Integral) or isinstance(nsim, (bool, np.bool_)) or nsim < 1:
        raise ValueError("nsim must be a positive integer")
    eta, gamma, log_shape = _linear_predictors(
        fit, groups, count_design, inflation_design, dispersion_design, offset,
    )
    levels, codes = _group_codes(groups)
    rng = np.random.default_rng(seed)
    offsets = rng.normal(scale=fit.random_intercept_sd, size=(nsim, len(levels)))
    linear = eta + offsets[:, codes]
    if (linear >= np.log(np.finfo(float).max)).any():
        raise ValueError("simulated count means exceed the supported numerical range")
    if log_shape is None:
        counts = rng.poisson(np.exp(linear))
    else:
        counts = rng.negative_binomial(np.exp(log_shape), expit(log_shape - linear))
    zeros = rng.random(counts.shape) < expit(gamma)
    counts[zeros] = 0
    return counts
