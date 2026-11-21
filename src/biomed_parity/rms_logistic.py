"""Quadratically penalized binary logistic regression in the style of rms::lrm."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit
from scipy.stats import chi2

from ._logistic import _deviance, _fit_coefficients, _information, _quadratic_penalty


@dataclass(frozen=True)
class PenalizedLogisticFit:
    """Intercept-first fit with covariance and effective degrees of freedom."""

    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    penalty_matrix: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    effective_df_diagonal: NDArray[np.float64]
    effective_df: float
    null_deviance: float
    deviance: float
    penalized_deviance: float
    likelihood_ratio: float
    pvalue: float


def _finite_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")
    return array.astype(float)


def _default_penalty(x, spread, categorical_blocks):
    matrix = np.diag(spread**2 * (len(x) / (len(x) - 1)))
    used = set()
    for block in categorical_blocks:
        indices = np.asarray(block)
        if (
            indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
            or np.any(indices >= x.shape[1]) or np.any(indices < 0)
            or len(set(indices)) != len(indices) or used.intersection(indices)
        ):
            raise ValueError("categorical_blocks must contain disjoint groups of valid column indices")
        indicators = x[:, indices]
        if np.any((indicators != 0) & (indicators != 1)) or np.any(indicators.sum(axis=1) > 1):
            raise ValueError("each categorical block must contain treatment-coded 0/1 indicators")
        used.update(indices)
        matrix[np.ix_(indices, indices)] = np.eye(len(indices)) - 1 / (len(indices) + 1)
    return matrix


def fit_penalized_logistic(
    x: ArrayLike,
    y: ArrayLike,
    *,
    penalty: ArrayLike = 0,
    penalty_matrix: ArrayLike | None = None,
    categorical_blocks: Sequence[Sequence[int]] = (),
) -> PenalizedLogisticFit:
    """Fit a binary logistic model from a finite slope design and 0/1 outcomes.

    The intercept is free of penalty. Pass a nonnegative penalty scalar or
    one factor per slope. The default base matrix uses sample variances,
    with optional treatment-coded categorical blocks identified by their
    zero-based column indices. A custom symmetric positive semidefinite
    base matrix can couple slopes. The effective penalty scales both axes
    by the square roots of the factors, as in rms::lrm.
    """
    x = _finite_array(x, "x")
    y = _finite_array(y, "y")
    if x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("x must be a nonempty slope design with one row per outcome")
    if np.any((y != 0) & (y != 1)) or y.min() == y.max():
        raise ValueError("y must contain both binary outcome values 0 and 1")
    factors = _finite_array(penalty, "penalty")
    p = x.shape[1]
    if factors.shape not in ((), (p,)) or np.any(factors < 0):
        raise ValueError("penalty must be nonnegative with one factor or one per slope")

    center = x.mean(axis=0)
    centered = x - center
    spread = centered.std(axis=0)
    if penalty_matrix is None:
        base = _default_penalty(x, spread, categorical_blocks)
    else:
        if len(categorical_blocks):
            raise ValueError("categorical_blocks apply only to the default penalty matrix")
        base = _finite_array(penalty_matrix, "penalty_matrix")
        if base.shape != (p, p) or not np.allclose(base, base.T, rtol=1e-12, atol=1e-15):
            raise ValueError("penalty_matrix must be symmetric with one row and column per slope")
        base = base * 0.5 + base.T * 0.5
        values = np.linalg.eigvalsh(base)
        if values[0] < -p * np.finfo(float).eps * np.max(np.abs(values)):
            raise ValueError("penalty_matrix must be positive semidefinite")

    root = np.broadcast_to(np.sqrt(factors), (p,))
    applied_penalty = (base * root[:, None]) * root
    spread = np.where(spread == 0, 1, spread)
    scaled_penalty = (applied_penalty / spread[:, None]) / spread
    design = np.column_stack((np.ones(len(y)), centered / spread))
    if np.linalg.matrix_rank(design) < p + 1:
        row_scale = np.max(np.abs(scaled_penalty), axis=1)
        penalty_directions = scaled_penalty / np.where(row_scale == 0, 1, row_scale)[:, None]
        identification = np.vstack((design, np.column_stack((np.zeros(p), penalty_directions))))
        if np.linalg.matrix_rank(identification) < p + 1:
            raise np.linalg.LinAlgError("the logistic design and penalty do not identify all coefficients")

    coefficients = _fit_coefficients(design, y, scaled_penalty if np.any(scaled_penalty) else None)
    linear = np.einsum("ij,j->i", design, coefficients)
    information = _information(design, linear)
    penalized_information = information.copy()
    penalized_information[1:, 1:] += scaled_penalty
    covariance = np.linalg.solve(penalized_information, np.eye(p + 1))
    # Centering changes the intercept only. Because the penalty excludes it,
    # each diagonal contribution is invariant to this scaling and centering.
    df_diagonal = np.einsum("ij,ji->i", information, covariance)
    effective_df = float(df_diagonal[1:].sum())
    to_raw_scale = np.diag(np.r_[1, 1 / spread])
    to_raw_scale[0, 1:] = -center / spread
    covariance_times_transform = np.einsum("ij,kj->ik", covariance, to_raw_scale)
    raw_covariance = np.einsum("ij,jk->ik", to_raw_scale, covariance_times_transform)
    raw_covariance = raw_covariance * 0.5 + raw_covariance.T * 0.5
    raw_coefficients = np.einsum("ij,j->i", to_raw_scale, coefficients)

    signed = 2 * y - 1
    deviance = _deviance(signed, linear)
    null_deviance = _deviance(signed, logit(y.mean()))
    likelihood_ratio = null_deviance - deviance
    pvalue = chi2.sf(likelihood_ratio, effective_df) if effective_df > 0 else 1.0
    return PenalizedLogisticFit(
        coefficients=raw_coefficients, covariance=raw_covariance, penalty_matrix=applied_penalty,
        linear_predictor=linear, probabilities=expit(linear), effective_df_diagonal=df_diagonal,
        effective_df=effective_df, null_deviance=float(null_deviance), deviance=float(deviance),
        penalized_deviance=float(deviance + _quadratic_penalty(coefficients[1:], scaled_penalty)),
        likelihood_ratio=float(likelihood_ratio), pvalue=float(pvalue),
    )
