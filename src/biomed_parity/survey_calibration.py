"""Linear and raking calibration of survey weights against known totals."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _encode_labels
from .survey_cox import _survey_covariance
from .survey_replicates import SurveyReplicateDesign


@dataclass(frozen=True)
class SurveyCalibrationResult:
    """Calibrated weights and benchmark agreement in the supplied column order."""

    weights: NDArray[np.float64]
    multipliers: NDArray[np.float64]
    control_totals: NDArray[np.float64]
    residuals: NDArray[np.float64]
    relative_errors: NDArray[np.float64]
    rank: int
    iterations: int
    converged: bool


@dataclass(frozen=True)
class SurveyCalibratedTotal:
    """Calibrated totals with one-stage Taylor covariance."""

    estimate: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    design_df: int
    calibration: SurveyCalibrationResult


def _numeric(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(float)


def _calibration_inputs(weights, variables, population, variance, epsilon):
    weights = _numeric(weights, "weights")
    if weights.ndim != 1 or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be a nonnegative vector with at least one positive value")
    matrix = _numeric(variables, "variables")
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != weights.size or matrix.shape[1] == 0:
        raise ValueError("variables must have one row per weight and at least one control column")
    population = np.atleast_1d(_numeric(population, "population"))
    if population.shape != (matrix.shape[1],):
        raise ValueError("population must have one total per control column")
    limits = _numeric(epsilon, "epsilon")
    if limits.shape not in ((), (1,), population.shape) or np.any(limits <= 0):
        raise ValueError("epsilon must be positive, with one value or one value per control")
    limits = np.broadcast_to(limits, population.shape)

    sigma2 = np.ones(weights.size)
    if variance is not None:
        sigma2 = np.atleast_1d(_numeric(variance, "variance"))
        if sigma2.shape != weights.shape:
            if sigma2.shape != population.shape:
                raise ValueError("variance must have one value per row or one coefficient per control")
            sigma2 = matrix @ sigma2
        if np.any(sigma2 <= 0) or not np.isfinite(sigma2).all():
            raise ValueError("variance must give a strictly positive finite value for each row")
    return weights, matrix, population, sigma2, limits


def _calibration_basis(weights, matrix, sigma2):
    positive = weights > 0
    supported = matrix[positive]
    # Scaling columns before factorization avoids dependence on units of measurement.
    scales = np.max(np.abs(supported), axis=0)
    scales = np.where(scales > 0, scales, 1)
    standardized = matrix / scales
    weighted = standardized[positive] * np.sqrt(weights[positive] / sigma2[positive])[:, None]
    _, singular_values, vectors = np.linalg.svd(weighted, full_matrices=False)
    cutoff = np.finfo(float).eps * max(weighted.shape) * singular_values[0]
    keep = singular_values > cutoff
    return standardized, scales, singular_values[keep], vectors[keep]


def _raking(weights, matrix, basis, population, scales, sigma2, limits, maxit):
    sample = matrix.T @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = population / (sample * scales)
    # survey rescales very small base weights before generalized raking.
    initial_scale = ratios.mean() if np.isfinite(ratios).all() and np.all(ratios > 20) else 1.0
    base = weights * initial_scale
    reduced = matrix @ basis.T
    total = population / scales
    relative_scale = scales / (1 + np.abs(population))
    target = basis @ total
    coefficients = np.zeros(basis.shape[0])
    multipliers = np.ones(weights.size)
    residual = total - matrix.T @ base
    if np.all(np.abs(residual) * relative_scale < limits):
        return multipliers * initial_scale, 0

    for iteration in range(1, maxit + 1):
        calibrated = base * multipliers
        gradient = target - reduced.T @ calibrated
        weighted = reduced * np.sqrt(calibrated / sigma2)[:, None]
        jacobian = weighted.T @ weighted
        step = np.linalg.lstsq(jacobian, gradient, rcond=256 * np.finfo(float).eps)[0]
        error = np.linalg.norm(residual * relative_scale)

        # Damping keeps exponentials finite and reduces population-total error.
        for _ in range(40):
            candidate = coefficients + step
            with np.errstate(over="ignore", invalid="ignore"):
                proposed = np.exp((reduced @ candidate) / sigma2)
                candidate_residual = total - matrix.T @ (base * proposed)
                candidate_error = np.linalg.norm(candidate_residual * relative_scale)
            if np.isfinite(candidate_error) and candidate_error < error:
                coefficients, multipliers, residual = candidate, proposed, candidate_residual
                break
            step *= 0.5
        else:
            break
        if np.all(np.abs(residual) * relative_scale < limits):
            break
    return multipliers * initial_scale, iteration


def calibrate(
    weights: ArrayLike,
    variables: ArrayLike,
    population: ArrayLike,
    *,
    calfun: str = "linear",
    variance: ArrayLike | None = None,
    epsilon: ArrayLike = 1e-7,
    maxit: int = 50,
    force: bool = False,
) -> SurveyCalibrationResult:
    """Calibrate existing sample weights using unbounded linear or raking distances.

    Supply one row per observation and one column per known population total.
    Include a column of ones to benchmark the population size. ``variance``
    specifies positive variance factors by row or coefficients evaluated on
    ``variables``. Raking allows consistent redundant controls. A control
    that is zero in the sample and in the population is redundant in either
    method. Raking stops when each absolute residual divided by
    ``1 + abs(population)`` is below ``epsilon``. ``force`` returns the last
    iterate with diagnostics if the requested accuracy was not achieved.
    """
    if calfun not in ("linear", "raking"):
        raise ValueError("calfun must be 'linear' or 'raking'")
    if not isinstance(maxit, Integral) or isinstance(maxit, (bool, np.bool_)) or maxit < 1:
        raise ValueError("maxit must be a positive integer")
    if not isinstance(force, (bool, np.bool_)):
        raise ValueError("force must be Boolean")
    weights, matrix, population, sigma2, limits = _calibration_inputs(
        weights, variables, population, variance, epsilon,
    )
    informative = np.any(matrix != 0, axis=0) | (population != 0)
    multipliers = np.ones(weights.size)
    rank = iterations = 0
    if informative.any():
        controls = population[informative]
        work, scales, singular, basis = _calibration_basis(weights, matrix[:, informative], sigma2)
        rank = singular.size
        if calfun == "linear" and rank != work.shape[1]:
            raise ValueError("linear calibration requires independent control columns with positive-weight support")
        if rank < work.shape[1]:
            normalized = controls / scales
            unsupported = (normalized - basis.T @ (basis @ normalized)) * scales
            if np.any(np.abs(unsupported) / (1 + np.abs(controls)) >= limits[informative]):
                raise ValueError("population controls are inconsistent with the positive-weight sample")

        if rank and calfun == "linear":
            discrepancy = controls / scales - work.T @ weights
            coefficients = basis.T @ ((basis @ discrepancy) / singular**2)
            multipliers = 1 + (work @ coefficients) / sigma2
            iterations = 1
        elif rank:
            multipliers, iterations = _raking(
                weights, work, basis, controls, scales, sigma2, limits[informative], maxit,
            )

    calibrated = weights * multipliers
    achieved = matrix.T @ calibrated
    residuals = achieved - population
    relative_errors = np.abs(residuals) / (1 + np.abs(population))
    converged = bool(np.all(relative_errors < limits))
    if not converged:
        message = f"calibration failed to converge (maximum relative control error {relative_errors.max():.6g})"
        if not force:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return SurveyCalibrationResult(calibrated, multipliers, achieved, residuals, relative_errors,
                                   rank, iterations, converged)


def calibrate_replicates(
    design: SurveyReplicateDesign,
    variables: ArrayLike,
    population: ArrayLike,
    *,
    calfun: str = "linear",
    variance: ArrayLike | None = None,
    epsilon: ArrayLike = 1e-7,
    maxit: int = 50,
) -> SurveyReplicateDesign:
    """Calibrate the full-sample weights and each replicate to the same totals."""
    if not isinstance(design, SurveyReplicateDesign):
        raise TypeError("design must be a SurveyReplicateDesign")
    settings = dict(calfun=calfun, variance=variance, epsilon=epsilon, maxit=maxit)
    sampling = calibrate(design.sampling_weights, variables, population, **settings)
    original = design.analysis_weights
    replicates = np.empty_like(original)
    for index, weights in enumerate(original.T):
        try:
            replicates[:, index] = calibrate(weights, variables, population, **settings).weights
        except (ValueError, RuntimeError) as error:
            raise type(error)(f"replicate {index + 1}: {error}") from error

    sampling.weights.setflags(write=False)
    replicates.setflags(write=False)
    calibrated = replace(
        design, sampling_weights=sampling.weights, replicate_weights=replicates, combined_weights=True,
        _df_override=design._df_override if calfun == "raking" else None,
    )
    if calfun == "raking":
        # survey retains the original full-design degrees of freedom for raking.
        # A later domain subset computes its own rank unless the user supplied df.
        calibrated.__dict__["design_df"] = design.design_df
    return calibrated


def calibrated_total(
    values: ArrayLike,
    weights: ArrayLike,
    variables: ArrayLike,
    population: ArrayLike,
    *,
    psu: ArrayLike | None = None,
    strata: ArrayLike | None = None,
    fpc: ArrayLike | None = None,
    calfun: str = "linear",
    variance: ArrayLike | None = None,
    epsilon: ArrayLike = 1e-7,
    maxit: int = 50,
) -> SurveyCalibratedTotal:
    """Estimate totals and GREG Taylor covariance for a one-stage design.

    Supply PSU labels, sampling strata and finite-population fractions by
    row as appropriate. The default treats each row as its own PSU in one
    stratum, with a negligible first-stage sampling fraction.
    """
    calibration = calibrate(weights, variables, population, calfun=calfun, variance=variance,
                            epsilon=epsilon, maxit=maxit)
    weights, controls, _, sigma2, _ = _calibration_inputs(weights, variables, population, variance, epsilon)
    n = weights.size
    data = _numeric(values, "values")
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] != n or data.shape[1] == 0:
        raise ValueError("values must have one row per sampled unit and at least one outcome")
    clusters = np.arange(n) if psu is None else _encode_labels(psu, None, n, "psu")[1]
    groups = np.zeros(n, dtype=int) if strata is None else _encode_labels(strata, None, n, "strata")[1]
    fractions = np.zeros(n) if fpc is None else _numeric(fpc, "fpc")
    if fractions.shape != (n,) or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("fpc must contain one sampling fraction in [0, 1] per observation")

    # R projects totals using the original design weights, with an additional
    # variance factor for unbounded linear calibration.
    projection_weights = np.sqrt(weights / sigma2) if calfun == "linear" else np.sqrt(weights)
    scales = np.max(np.abs(controls[weights > 0]), axis=0)
    scaled_controls = controls / np.where(scales > 0, scales, 1)
    coefficients = np.linalg.lstsq(
        scaled_controls * projection_weights[:, None], data * projection_weights[:, None], rcond=None,
    )[0]
    residuals = data - scaled_controls @ coefficients
    covariance, design_df = _survey_covariance(calibration.weights[:, None] * residuals, groups, clusters, fractions)
    return SurveyCalibratedTotal(calibration.weights @ data, covariance, np.sqrt(np.diag(covariance)),
                                 design_df, calibration)
