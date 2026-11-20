"""Predict study effects from a fitted multilevel meta-regression."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, t

from .meta_multilevel import MetaMultilevelResult, _real_array


@dataclass(frozen=True)
class MetaPrediction:
    """Means, true-effect prediction intervals and row-wise precision."""

    estimate: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    confidence_intervals: NDArray[np.float64]
    prediction_standard_errors: NDArray[np.float64]
    prediction_intervals: NDArray[np.float64]
    degrees_of_freedom: NDArray[np.float64] | None
    prediction_degrees_of_freedom: NDArray[np.float64] | None
    covariance: NDArray[np.float64] | None


def _row_variance(value, rows, name):
    array = _real_array(value, name)
    if array.shape not in ((), (rows,)) or (array < 0).any():
        raise ValueError(f"{name} must be a nonnegative scalar or have one value per design row")
    return np.broadcast_to(array, (rows,))


def _prediction_design(fit, design):
    if design is None:
        return fit._design
    rows = _real_array(design, "design")
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or not len(rows) or rows.shape[1] != len(fit.coefficients):
        raise ValueError("design must contain one column per fitted coefficient, including any intercept")
    return rows


def _prediction_degrees(fit, rows):
    degrees = fit.inference.degrees_of_freedom
    if np.isnan(degrees).all():
        return None
    if np.all(degrees == degrees[0]):
        return np.full(len(rows), degrees[0])
    return np.where(rows != 0, degrees[None, :], np.inf).min(axis=1)


def _interval(estimate, standard_errors, critical):
    half_width = standard_errors * critical
    return np.column_stack((estimate - half_width, estimate + half_width))


def predict_multilevel_meta(
    fit: MetaMultilevelResult,
    design: ArrayLike | None = None,
    *,
    heterogeneity: ArrayLike | None = None,
    new_sampling_variance: ArrayLike = 0.0,
    confidence_level: float | None = None,
    adjust: bool = False,
    prediction: Literal["default", "simple", "t", "riley"] = "default",
    transform: Callable[[float], float] | None = None,
    covariance: bool = False,
) -> MetaPrediction:
    """Predict with design columns in the fitted coefficient order.

    The supplied design includes any intercept. With no design, reuse the
    fitted rows. The prediction interval targets a new true effect unless
    optional new_sampling_variance is supplied. If fitted heterogeneity
    differs by row, pass new heterogeneity explicitly for new designs.
    """
    if not isinstance(fit, MetaMultilevelResult):
        raise ValueError("fit must be a MetaMultilevelResult")
    if prediction not in ("default", "simple", "t", "riley"):
        raise ValueError("prediction must be default, simple, t, or riley")
    if not all(isinstance(value, (bool, np.bool_)) for value in (adjust, covariance)):
        raise ValueError("adjust and covariance must be boolean")
    level = fit.confidence_level if confidence_level is None else confidence_level
    if not isinstance(level, Real) or not 0 < level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if transform is not None and not callable(transform):
        raise ValueError("transform must be callable")
    rows = _prediction_design(fit, design)
    m = len(rows)

    if heterogeneity is None:
        random_variance = fit._random_variance
        if random_variance is None:
            heterogeneity = sum(fit.variance_components.values())
        elif design is None:
            heterogeneity = random_variance
        elif np.allclose(random_variance, random_variance[0], rtol=1e-12, atol=0):
            heterogeneity = random_variance[0]
        else:
            raise ValueError("give heterogeneity for a new design with nonconstant fitted random variance")
    tau = _row_variance(heterogeneity, m, "heterogeneity")
    new_variance = _row_variance(new_sampling_variance, m, "new_sampling_variance")
    estimates = rows @ fit.coefficients
    weighted = rows @ fit.inference.covariance
    fitted_variance = np.maximum(np.einsum("ij,ij->i", weighted, rows), 0)
    se = np.sqrt(fitted_variance)
    alpha = (1 - level) / (2 * (m if adjust else 1))
    degrees = _prediction_degrees(fit, rows)
    critical = norm.isf(alpha) if degrees is None else t.isf(alpha, degrees)
    ci = _interval(estimates, se, critical)

    prediction_degrees = degrees
    prediction_variance = fitted_variance
    if prediction == "simple":
        prediction_degrees = None
        prediction_variance = 0
        critical = norm.isf(alpha)
    elif prediction in ("t", "riley"):
        if prediction_degrees is None:
            residual = len(fit._design) - len(fit.coefficients)
            prediction_degrees = np.full(m, residual, dtype=float)
        if prediction == "riley":
            estimated = fit._estimated_variances
            if estimated is None:
                estimated = len(fit.variance_components)
            prediction_degrees = np.maximum(prediction_degrees - estimated, 1)
        critical = t.isf(alpha, prediction_degrees)
    pi_se = np.sqrt(prediction_variance + tau + new_variance)
    pi = _interval(estimates, pi_se, critical)
    prediction_covariance = weighted @ rows.T if covariance and transform is None else None
    if prediction_covariance is not None:
        prediction_covariance = (prediction_covariance + prediction_covariance.T) / 2
    if transform is not None:
        def transformed(array):
            return _real_array([transform(value) for value in array.ravel()], "transformed values").reshape(array.shape)

        estimates = transformed(estimates)
        ci, pi = np.sort(transformed(ci), axis=1), np.sort(transformed(pi), axis=1)
        se = np.full(m, np.nan)
    return MetaPrediction(estimates, se, ci, pi_se, pi, degrees, prediction_degrees, prediction_covariance)
