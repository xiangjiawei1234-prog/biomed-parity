"""Case-bootstrap inference for semiparametric interval regression."""

from dataclasses import dataclass
from math import ceil, fsum
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from .interval_regression import IntervalRegressionFit, fit_interval_regression


@dataclass(frozen=True)
class IntervalRegressionBootstrap:
    """Original fit, successful bootstrap slopes, and Wald inference."""

    fit: IntervalRegressionFit
    replicates: NDArray[np.float64]
    failed_replicates: tuple[int, ...]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    z: NDArray[np.float64]
    pvalue: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


def _resample_counts(weights, bootstrap_indices, n_resamples, seed):
    n = len(weights)
    total = fsum(weights)
    draw_size = ceil(total)
    positive = weights > 0
    if bootstrap_indices is not None:
        if n_resamples is not None or seed is not None:
            raise ValueError("n_resamples and seed apply when bootstrap_indices is omitted")
        indices = np.asarray(bootstrap_indices)
        if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] != draw_size:
            raise ValueError("bootstrap_indices must have one row per replicate and ceil(sum(weights)) columns")
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("bootstrap_indices must contain zero-based integer row indices")
        if not positive[indices].all():
            raise ValueError("bootstrap_indices cannot select zero-weight rows")
        return len(indices), (np.bincount(row.astype(np.intp, copy=False), minlength=n) for row in indices)

    count = 200 if n_resamples is None else n_resamples
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
        raise ValueError("n_resamples must be a positive integer")
    if draw_size > np.iinfo(np.int64).max:
        raise ValueError("bootstrap draw size exceeds the supported range")
    rng = np.random.default_rng(seed)
    probabilities = weights[positive] / total

    def draw_counts():
        for _ in range(count):
            multiplicity = np.zeros(n, dtype=np.int64)
            multiplicity[positive] = rng.multinomial(draw_size, probabilities)
            yield multiplicity

    return count, draw_counts()


def bootstrap_interval_regression(
    left: ArrayLike,
    right: ArrayLike,
    moderators: ArrayLike,
    *,
    model: Literal["ph", "po"] = "ph",
    bounds: tuple[bool, bool] = (False, True),
    weights: ArrayLike | None = None,
    bootstrap_indices: ArrayLike | None = None,
    n_resamples: int | None = None,
    seed: int | None = None,
    confidence_level: float = 0.95,
) -> IntervalRegressionBootstrap:
    """Refit case bootstrap samples and estimate slope covariance and Wald intervals.

    Each draw samples with replacement in proportion to the case weights.
    There are ceil(sum(weights)) draws in each replicate, and their counts
    become the new case weights. Supply explicit zero-based row indices or
    generate n_resamples (200 by default) with a NumPy random seed.

    The original coefficients center the Wald intervals. Covariance uses
    the sample covariance of the successful estimates. Fewer than two
    successes yield missing inference, with the original fit still returned.
    """
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    left, right, moderators = np.asarray(left), np.asarray(right), np.asarray(moderators)
    fit = fit_interval_regression(left, right, moderators, model=model, bounds=bounds, weights=weights)
    case_weights = np.ones(len(left)) if weights is None else np.asarray(weights, dtype=float)
    count, samples = _resample_counts(case_weights, bootstrap_indices, n_resamples, seed)

    size = len(fit.coefficients)
    estimates = np.empty((count, size))
    failures = []
    successful = 0
    for i, multiplicity in enumerate(samples):
        # Reversing the summation order can recover a numerical optimization
        # failure without changing the rows, weights, or likelihood.
        for step in (1, -1):
            try:
                refit = fit_interval_regression(
                    left[::step], right[::step], moderators[::step],
                    model=model, bounds=bounds, weights=multiplicity[::step],
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(refit.coefficients).all():
                estimates[successful] = refit.coefficients
                successful += 1
                break
        else:
            failures.append(i)

    estimates = estimates[:successful].copy()
    if successful < 2:
        covariance = np.full((size, size), np.nan)
    else:
        centered = estimates - estimates.mean(axis=0)
        covariance = np.einsum("ij,ik->jk", centered, centered) / (successful - 1)
    standard_error = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = fit.coefficients / standard_error
    pvalue = 2 * ndtr(-np.abs(z))
    margin = -ndtri((1 - confidence_level) / 2) * standard_error
    return IntervalRegressionBootstrap(
        fit, estimates, tuple(failures), covariance, standard_error, z, pvalue,
        fit.coefficients - margin, fit.coefficients + margin,
    )
