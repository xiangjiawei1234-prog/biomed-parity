"""Gaussian and binary logistic GLMs with replicate-weight survey covariance."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit
from scipy.stats import t as student_t

from .survey_replicates import SurveyReplicateDesign, _replicate_covariance


@dataclass(frozen=True)
class SurveyGLMResult:
    """Full-sample GLM coefficients, replicate inference, and model diagnostics."""

    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    naive_covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    t_statistics: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    design_df: float
    residual_df: float
    inference_df: float
    deviance: float
    dispersion: float
    fitted_values: NDArray[np.float64]
    replicate_estimates: NDArray[np.float64]
    variance_center: NDArray[np.float64]
    included: NDArray[np.bool_]


class _FitError(ValueError):
    """A weighted fit cannot produce a full-rank, finite coefficient vector."""


def _numeric(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be numeric")
    return array.astype(float)


def _weighted_least_squares(matrix, target, weights, tolerance):
    root_weight = np.sqrt(weights)
    weighted = matrix * root_weight[:, None]
    column_norms = np.linalg.norm(weighted, axis=0)
    if np.any(column_norms == 0):
        raise _FitError("the positive-weight model matrix is rank deficient")
    left, singular, right = np.linalg.svd(weighted / column_norms, full_matrices=False)
    if singular.size != matrix.shape[1] or singular[-1] <= tolerance * singular[0]:
        raise _FitError("the positive-weight model matrix is rank deficient")
    axes = right.T / column_norms[:, None] / singular
    weighted_target = root_weight * target
    coordinates = np.einsum("ij,i->j", left, weighted_target)
    coefficients = np.einsum("ij,j->i", axes, coordinates)
    if not np.isfinite(coefficients).all():
        raise _FitError("weighted fit produced nonfinite coefficients")
    return coefficients, axes @ axes.T


def _linear_predictor(matrix, coefficients, offset):
    return np.einsum("ij,j->i", matrix, coefficients) + offset


def _binary_deviance(response, eta, weights):
    loss = np.logaddexp(0, (1 - 2 * response) * eta)
    return 2 * np.sum(weights * loss)


def _weighted_fit(matrix, response, weights, offset, family, epsilon, maxit, start=None):
    inform = weights > 0
    x, y, w, off = matrix[inform], response[inform], weights[inform], offset[inform]
    if x.shape[0] < x.shape[1]:
        raise _FitError("too few positive-weight rows for the model")
    tolerance = min(1e-7, epsilon / 1000)
    if family == "gaussian":
        coefficients, naive = _weighted_least_squares(x, y - off, w, tolerance)
        fitted = _linear_predictor(matrix, coefficients, offset)
        deviance = np.sum(w * (y - fitted[inform]) ** 2)
        return coefficients, naive, fitted, deviance, deviance

    if start is None:
        # R's binary GLM starts from response-specific means corrected for prior weights.
        means = (w * y + 0.5) / (w + 1)
        eta = logit(np.clip(means, np.finfo(float).eps, 1 - np.finfo(float).eps))
    else:
        eta = _linear_predictor(x, start, off)
    deviance = _binary_deviance(y, eta, w)
    for _ in range(maxit):
        variance = expit(eta) * expit(-eta)
        if np.any(variance <= np.finfo(float).eps):
            raise _FitError("binary logistic fit reached a numerical boundary")
        # Write (y - p) by response group to retain precision in the tails.
        difference = np.where(y == 1, expit(-eta), -expit(eta))
        target = eta - off + difference / variance
        coefficients, naive = _weighted_least_squares(x, target, w * variance, tolerance)
        eta = _linear_predictor(x, coefficients, off)
        current = _binary_deviance(y, eta, w)
        if not np.isfinite(current):
            raise _FitError("binary logistic fit produced nonfinite deviance")
        if abs(current - deviance) / (0.1 + current) < epsilon:
            full_eta = _linear_predictor(matrix, coefficients, offset)
            fitted = expit(full_eta)
            variance = expit(eta) * expit(-eta)
            if np.any(variance <= np.finfo(float).eps):
                raise _FitError("binary logistic fit reached a numerical boundary")
            difference = np.where(y == 1, expit(-eta), -expit(eta))
            pearson = np.sum(w * (difference**2 / variance))
            return coefficients, naive, fitted, current, pearson
        deviance = current
    raise _FitError("binary logistic fit did not converge")


def _model_inputs(values, covariates, offset, n, intercept):
    values = _numeric(values, "values")
    matrix = np.empty((n, 0)) if covariates is None else _numeric(covariates, "covariates")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    offsets = np.zeros(n) if offset is None else _numeric(offset, "offset")
    if values.shape != (n,) or matrix.ndim != 2 or matrix.shape[0] != n or offsets.shape != (n,):
        raise ValueError("values, covariates and offset must have one row per design observation")
    if intercept:
        matrix = np.column_stack((np.ones(n), matrix))
    if matrix.shape[1] == 0:
        raise ValueError("the model must have at least one coefficient")
    return values, matrix, offsets


def svyglm(
    values: ArrayLike,
    covariates: ArrayLike | None,
    design: SurveyReplicateDesign,
    *,
    family: str = "gaussian",
    intercept: bool = True,
    offset: ArrayLike | None = None,
    subset: ArrayLike | None = None,
    na_rm: bool = True,
    rescale: bool | None = None,
    df_resid: float | None = None,
    confidence_level: float = 0.95,
    epsilon: float = 1e-8,
    maxit: int = 25,
) -> SurveyGLMResult:
    """Fit a replicate-weight Gaussian identity or binary quasibinomial logit GLM.

    Outcomes and numeric covariates are aligned with the design's rows.
    An intercept is added first unless disabled. Use ``covariates=None``
    for an intercept-only fit, ``subset`` for a Boolean domain mask, and
    ``offset`` for known contributions on the linear-predictor scale.
    Missing model rows are dropped unless ``na_rm=False``. Coefficient
    inference uses survey's residual degrees of freedom or ``df_resid``.
    ``rescale`` controls the full-sample prior weights used in diagnostics.
    """
    if not isinstance(design, SurveyReplicateDesign):
        raise TypeError("design must be a SurveyReplicateDesign")
    if family not in ("gaussian", "quasibinomial"):
        raise ValueError("family must be 'gaussian' or 'quasibinomial'")
    if not all(isinstance(flag, (bool, np.bool_)) for flag in (intercept, na_rm)) or (
        rescale is not None and not isinstance(rescale, (bool, np.bool_))
    ):
        raise ValueError("intercept, na_rm and rescale must be Boolean when supplied")
    if not isinstance(maxit, Integral) or isinstance(maxit, (bool, np.bool_)) or maxit < 1:
        raise ValueError("maxit must be a positive integer")
    if not isinstance(epsilon, Real) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if df_resid is not None and (not isinstance(df_resid, Real) or np.isnan(df_resid) or df_resid <= 0):
        raise ValueError("df_resid must be positive or infinity")

    n = design.sampling_weights.size
    y, x, off = _model_inputs(values, covariates, offset, n, intercept)
    selected = np.ones(n, dtype=bool)
    if subset is not None:
        selected = np.asarray(subset)
        design = design.subset(selected)
        y, x, off = y[selected], x[selected], off[selected]
    if np.isinf(y).any() or np.isinf(x).any() or np.isinf(off).any():
        raise ValueError("values, covariates and offset must not contain infinities")
    present = ~(np.isnan(y) | np.isnan(x).any(axis=1) | np.isnan(off))
    if not present.all() and not na_rm:
        raise ValueError("model inputs contain missing values; use na_rm=True to remove incomplete rows")
    if not present.any():
        raise ValueError("no complete rows remain for the model")
    y, x, off = y[present], x[present], off[present]
    included = np.zeros(n, dtype=bool)
    included[np.flatnonzero(selected)[present]] = True
    if family == "quasibinomial" and not np.isin(y, (0, 1)).all():
        raise ValueError("quasibinomial values must be binary (0 or 1)")

    raw_weights = design.sampling_weights
    replicate_weights = design.analysis_weights[present]
    if np.any(raw_weights < 0) or np.any(replicate_weights < 0):
        raise ValueError("sampling and replicate analysis weights must be nonnegative")
    full_weights = raw_weights[present]
    if not np.any(full_weights > 0):
        raise ValueError("the full sample must have positive weight on some complete rows")
    if rescale is None:
        full_weights = full_weights / raw_weights.mean()
    elif rescale:
        full_weights = full_weights / raw_weights.sum()

    coefficients, naive, fitted, deviance, pearson = _weighted_fit(
        x, y, full_weights, off, family, epsilon, maxit,
    )
    # R passes the initialized binary response from the full fit to every replicate.
    replicate_response = np.where(full_weights == 0, 0, y) if family == "quasibinomial" else y
    replicates = np.full((replicate_weights.shape[1], coefficients.size), np.nan)
    for i, weights in enumerate(replicate_weights.T):
        if np.any(weights > 0):
            try:
                replicates[i] = _weighted_fit(
                    x, replicate_response, weights / weights.sum(), off, family, epsilon, maxit, start=coefficients,
                )[0]
            except _FitError:
                continue
    valid = np.isfinite(replicates).all(axis=1)
    if not valid.any():
        raise ValueError("no replicate produced a finite full-rank fit")
    if not valid.all():
        warnings.warn(
            f"{np.count_nonzero(~valid)} invalid replicate fits were discarded", RuntimeWarning, stacklevel=2,
        )
    covariance, center = _replicate_covariance(
        replicates[valid], coefficients, design.scale, design.rscales[valid], design.mse,
    )
    standard_errors = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = coefficients / standard_errors
    design_df = design.design_df
    residual_df = design_df + 1 - coefficients.size
    inference_df = residual_df if df_resid is None else float(df_resid)
    pvalues = 2 * student_t.sf(np.abs(statistics), inference_df)
    margin = student_t.isf((1 - confidence_level) / 2, inference_df) * standard_errors
    # In summary.svrepglm, the survey weights cancel the Pearson residual scaling.
    dispersion = pearson / y.size
    return SurveyGLMResult(
        coefficients, covariance, naive, standard_errors, statistics, pvalues, coefficients - margin,
        coefficients + margin, design_df, residual_df, inference_df, float(deviance), float(dispersion),
        fitted, replicates, center, included,
    )
