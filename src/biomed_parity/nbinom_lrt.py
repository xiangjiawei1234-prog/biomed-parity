"""Likelihood-ratio tests for nested negative-binomial count models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import qr
from scipy.special import expit, gammaln
from scipy.stats import chi2


@dataclass(frozen=True)
class NegativeBinomialLRT:
    statistic: NDArray[np.float64]
    pvalue: NDArray[np.float64]
    adjusted_pvalue: NDArray[np.float64]
    full_log_likelihood: NDArray[np.float64]
    reduced_log_likelihood: NDArray[np.float64]
    tested: NDArray[np.bool_]
    degrees_of_freedom: int


def _array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be real numeric")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _orthogonal_design(values, samples, name):
    design = _array(values, name)
    if design.ndim != 2 or design.shape[0] != samples or not 0 < design.shape[1] < samples:
        raise ValueError(f"{name} must be a sample-aligned matrix with positive residual df")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"{name} must have full column rank")
    q_matrix, _ = qr(design, mode="economic")
    return q_matrix


def _log_likelihood(design, observations, offset, size):
    def evaluate(coefficients):
        log_odds = (design @ coefficients) + offset - np.log(size)
        log_likelihood = (
            -observations @ np.logaddexp(0, -log_odds)
            - size * np.sum(np.logaddexp(0, log_odds))
        )
        positive = expit(log_odds)
        negative = expit(-log_odds)
        # The two terms stay finite at small fitted means and large size.
        score = design.T @ (observations * negative - size * positive)
        curvature = (observations + size) * positive * negative
        information = design.T @ (design * curvature[:, None])
        return log_likelihood, score, information

    initial = np.log(observations + 0.5) - offset
    coefficients = design.T @ initial
    log_likelihood, score, information = evaluate(coefficients)
    for _ in range(100):
        if np.max(np.abs(score)) < 1e-9:
            return log_likelihood
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular information in the negative-binomial fit") from exc
        for step_size in 2.0 ** -np.arange(32):
            candidate = coefficients + step_size * step
            value, next_score, next_information = evaluate(candidate)
            if value >= log_likelihood - 1e-12:
                break
        else:
            raise ValueError("negative-binomial likelihood failed to increase")
        coefficients = candidate
        log_likelihood, score, information = value, next_score, next_information
        if np.max(np.abs(step_size * step)) < 1e-10:
            return log_likelihood
    raise ValueError("negative-binomial model did not converge")


def _adjust_finite(pvalues):
    finite = np.flatnonzero(np.isfinite(pvalues))
    order = finite[np.argsort(pvalues[finite], kind="stable")]
    values = len(order) * pvalues[order] / np.arange(1, len(order) + 1)
    adjusted = np.full_like(pvalues, np.nan)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(values[::-1])[::-1])
    return adjusted


def nbinom_lrt(
    counts: ArrayLike,
    full: ArrayLike,
    reduced: ArrayLike,
    dispersions: ArrayLike,
    size_factors: ArrayLike,
) -> NegativeBinomialLRT:
    """Fit nested log-link NB2 models with fixed dispersions and offsets.

    Rows of counts index features, columns index samples. Design rows are
    sample-aligned. A factor per sample or a feature-by-sample normalization
    matrix supplies multiplicative offsets. All-zero features are omitted.
    """
    observations = _array(counts, "counts")
    if (
        observations.ndim != 2 or not observations.shape[0]
        or np.any((observations < 0) | (observations != np.floor(observations)))
    ):
        raise ValueError("counts must be a feature-by-sample matrix of nonnegative integers")
    features, samples = observations.shape
    full_design = _orthogonal_design(full, samples, "full")
    reduced_design = _orthogonal_design(reduced, samples, "reduced")
    df = full_design.shape[1] - reduced_design.shape[1]
    projected = full_design @ (full_design.T @ reduced_design)
    if df <= 0 or not np.allclose(projected, reduced_design, rtol=0, atol=1e-10):
        raise ValueError("reduced must be strictly nested within full")

    normalization = _array(size_factors, "size_factors")
    if normalization.shape not in ((samples,), observations.shape) or np.any(normalization <= 0):
        raise ValueError("size_factors must be positive sample- or feature-by-sample factors")
    offset = np.broadcast_to(np.log(normalization), observations.shape)
    tested = np.any(observations != 0, axis=1)
    dispersion_array = np.asarray(dispersions)
    if dispersion_array.shape != (features,) or dispersion_array.dtype.kind not in "buif":
        raise ValueError("dispersions must supply a positive value for each tested feature")
    dispersion_array = dispersion_array.astype(float)
    if not np.all(np.isfinite(dispersion_array[tested]) & (dispersion_array[tested] > 0)):
        raise ValueError("dispersions must supply a positive value for each tested feature")

    full_ll = np.full(features, np.nan)
    reduced_ll = np.full(features, np.nan)
    statistic = np.full(features, np.nan)
    for feature in np.flatnonzero(tested):
        y = observations[feature]
        theta = 1 / dispersion_array[feature]
        if not np.isfinite(theta):
            raise ValueError(f"dispersion for feature {feature} is too small")
        try:
            reduced_value = _log_likelihood(reduced_design, y, offset[feature], theta)
            full_value = _log_likelihood(full_design, y, offset[feature], theta)
        except ValueError as exc:
            raise ValueError(f"fit failed for feature {feature}: {exc}") from exc
        constant = np.sum(gammaln(y + theta) - gammaln(theta) - gammaln(y + 1))
        reduced_ll[feature] = reduced_value + constant
        full_ll[feature] = full_value + constant
        statistic[feature] = max(0, 2 * (full_value - reduced_value))

    pvalues = chi2.sf(statistic, df)
    return NegativeBinomialLRT(
        statistic, pvalues, _adjust_finite(pvalues), full_ll, reduced_ll, tested, df,
    )
