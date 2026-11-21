"""Maximum likelihood regression for the Royston–Parmar survival model."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import erfcx, log_ndtr

from .interval_censoring import _observation_intervals
from .royston_parmar import (
    RoystonParmarResult, RoystonParmarScale, TimeScale, _DEFAULT_KNOTS, _LOG_MILLS_FACTOR,
    _LOG_TWO, _SQRT_TWO, _broadcast_inputs, _link_values, _log_complement,
    _real_array, _spline_at, _spline_terms, royston_parmar,
)


@dataclass(frozen=True)
class RoystonParmarFit:
    """Estimates in baseline, location, then ascending ancillary-index order."""

    gamma: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    ancillary_coefficients: dict[int, NDArray[np.float64]]
    covariance: NDArray[np.float64] | None
    knots: NDArray[np.float64]
    scale: RoystonParmarScale
    timescale: TimeScale
    log_likelihood: float
    n_observations: int

    def predict(
        self, time: ArrayLike, design: ArrayLike | None = None, *,
        ancillary: Mapping[int, ArrayLike] | None = None, entry: ArrayLike | None = None,
    ) -> RoystonParmarResult:
        """Predict distribution values at supplied profile rows, optionally after entry."""
        ancillary = {} if ancillary is None else ancillary
        if not isinstance(ancillary, Mapping) or set(ancillary) - set(self.ancillary_coefficients):
            raise ValueError("ancillary prediction inputs must use fitted gamma indices")
        location = _prediction_effect(design, self.coefficients, "design")
        effects = {j: _prediction_effect(ancillary.get(j), beta, f"ancillary[{j}]")
            for j, beta in self.ancillary_coefficients.items()}
        shape = np.broadcast_shapes(np.shape(location), *(np.shape(effect) for effect in effects.values()))
        gamma = np.broadcast_to(self.gamma, shape + self.gamma.shape).copy()
        gamma[..., 0] += location
        for j, effect in effects.items():
            gamma[..., j] += effect
        options = dict(knots=self.knots, scale=self.scale, timescale=self.timescale)
        if entry is None:
            return royston_parmar(time, gamma, **options)

        entry = _real_array(entry, "entry")
        if not np.isfinite(entry).all() or np.any(entry < 0):
            raise ValueError("prediction entry times must be finite and nonnegative")
        time, entry = np.broadcast_arrays(_real_array(time, "time"), entry)
        result = royston_parmar(time, gamma, **options)
        initial = royston_parmar(entry, gamma, **options).log_survival
        if not np.isfinite(initial).all():
            raise ValueError("conditional predictions require a positive probability of surviving to entry")
        before = time < entry
        log_survival = np.where(before, 0, result.log_survival - initial)
        log_density = np.where(before, -np.inf, result.log_density - initial)
        log_hazard = np.where(before, -np.inf, result.log_hazard)
        with np.errstate(over="ignore"):
            return RoystonParmarResult(np.exp(log_survival), -log_survival, np.exp(log_hazard),
                np.exp(log_density), -np.expm1(log_survival), log_survival, log_hazard, log_density)


def _prediction_effect(design, coefficients, name):
    if design is None:
        return 0.0
    result = _real_array(design, name)
    if result.ndim == 0 or result.shape[-1] != coefficients.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain one finite value per fitted coefficient in its final axis")
    return result @ coefficients


class _ObservationIntervals(NamedTuple):
    left: NDArray[np.float64]
    right: NDArray[np.float64]
    entry: NDArray[np.float64]
    exact: NDArray[np.bool_]
    weights: NDArray[np.float64]
    keep: NDArray[np.bool_]


def _survival_intervals(left, right, entry, weights):
    lower, upper, _, weights, keep = _observation_intervals(left, right, (False, True), weights)
    if entry is None:
        entry = np.zeros(lower.shape)
    else:
        try:
            entry = np.broadcast_to(_real_array(entry, "entry"), keep.shape)[keep]
        except ValueError as error:
            raise ValueError("entry must be scalar or contain one time per observation") from error
    exact = lower == upper
    if (not np.isfinite(entry).all() or np.any(entry < 0)
        or np.any((entry >= upper) & np.isfinite(upper)) or np.any(entry[np.isinf(upper)] > lower[np.isinf(upper)])):
        raise ValueError("entry must precede the event or interval upper bound and not exceed a censoring time")
    return _ObservationIntervals(np.maximum(lower, entry), upper, entry, exact, weights, keep)


class _LinkTerms(NamedTuple):
    log_survival: NDArray[np.float64]
    log_cdf: NDArray[np.float64]
    log_density: NDArray[np.float64]
    survival_score: NDArray[np.float64]
    cdf_score: NDArray[np.float64]
    density_score: NDArray[np.float64]


def _link_terms(eta, scale):
    area, log_hazard, log_density = _link_values(eta, scale)
    if scale == "hazard":
        log_cdf = np.where(eta < -36, eta, _log_complement(-area))
        log_cdf_score = log_density - log_cdf
        density_score = 1 - area
    elif scale == "odds":
        log_cdf = log_hazard
        log_cdf_score = -area
        density_score = 1 - 2 * np.exp(log_cdf)
    else:
        log_cdf = log_ndtr(eta)
        log_cdf_score = np.where(eta < 0, _LOG_MILLS_FACTOR - np.log(erfcx(np.abs(eta) / _SQRT_TWO)),
            log_density - log_cdf)
        density_score = -eta
    finite = np.isfinite(eta)
    return _LinkTerms(-area, log_cdf, log_density, np.where(finite, -np.exp(log_hazard), 0),
        np.where(finite, np.exp(log_cdf_score), 0), density_score)


def _log_likelihood_terms(eta_low, eta_high, eta_entry, sloped, exact, log_jacobian, scale):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        low, high, entry = (_link_terms(eta, scale) for eta in (eta_low, eta_high, eta_entry))
        use_cdf = high.log_cdf < -_LOG_TWO
        larger = np.where(use_cdf, high.log_cdf, low.log_survival)
        smaller = np.where(use_cdf, low.log_cdf, high.log_survival)
        delta = np.minimum(smaller - larger, 0)
        log_difference = larger + _log_complement(delta)
        above = -np.expm1(delta)
        below = np.exp(delta)
        score_low = np.where(use_cdf, -low.cdf_score * below, low.survival_score) / above
        score_high = np.where(use_cdf, high.cdf_score, -high.survival_score * below) / above

        log_difference[exact] = (low.log_density[exact] + np.log(sloped[exact]) + log_jacobian[exact])
        score_low[exact] = low.density_score[exact]
        score_high[exact] = 0
        score_slope = np.divide(1, sloped, out=np.zeros_like(sloped), where=exact)
        value = log_difference - entry.log_survival
    return value, score_low, score_high, -entry.survival_score, score_slope


def _score_at(time, gamma, knots, timescale):
    positive = (time > 0) & np.isfinite(time)
    x = np.log(time[positive]) if timescale == "log" else time[positive]
    eta = np.where(time == 0, -np.inf, np.inf)
    slope = np.zeros_like(time)
    eta[positive], slope[positive] = _spline_at(x, gamma[positive], knots[positive])
    return eta, slope


def _jacobian(intervals, timescale):
    jacobian = np.zeros_like(intervals.left)
    if timescale == "log":
        jacobian[intervals.exact] = -np.log(intervals.left[intervals.exact])
    return jacobian


def royston_parmar_log_likelihood(
    left: ArrayLike, right: ArrayLike, gamma: ArrayLike, *, knots: ArrayLike = _DEFAULT_KNOTS,
    scale: RoystonParmarScale = "hazard", timescale: TimeScale = "log",
    entry: ArrayLike | None = None, weights: ArrayLike | None = None,
) -> float:
    """Weighted likelihood for exact events and left, right, or interval censoring.

    Equal endpoints are exact events, zero lower bounds indicate left
    censoring, and infinite upper bounds indicate right censoring. Entry
    conditions each observation on survival past its entry time.
    """
    intervals = _survival_intervals(left, right, entry, weights)
    _, gamma, knots = _broadcast_inputs(np.zeros(intervals.keep.shape), gamma, knots, scale, timescale)
    if gamma.ndim != 2:
        raise ValueError("gamma and knots require one row per observation or one common vector")
    gamma, knots = gamma[intervals.keep], knots[intervals.keep]
    with np.errstate(over="ignore", invalid="ignore"):
        lower, slopes = _score_at(intervals.left, gamma, knots, timescale)
        upper, _ = _score_at(intervals.right, gamma, knots, timescale)
        at_entry, _ = _score_at(intervals.entry, gamma, knots, timescale)
        values, *_ = _log_likelihood_terms(lower, upper, at_entry, slopes, intervals.exact,
            _jacobian(intervals, timescale), scale)
    return float(intervals.weights @ values) if np.isfinite(values).all() else -np.inf


def _design_matrix(design, keep, name):
    if design is None:
        return np.empty((np.count_nonzero(keep), 0))
    matrix = _real_array(design, name)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) != len(keep) or not np.isfinite(matrix[keep]).all():
        raise ValueError(f"{name} must contain one finite design row per observation")
    matrix = matrix[keep]
    if np.linalg.matrix_rank(np.column_stack((np.ones(len(matrix)), matrix))) != matrix.shape[1] + 1:
        raise ValueError(f"{name} must be full rank and exclude an intercept")
    return matrix


def _time_basis(time, knots, timescale):
    positive = (time > 0) & np.isfinite(time)
    x = np.log(time[positive]) if timescale == "log" else time[positive]
    basis = np.zeros((len(time), knots.size))
    derivative = np.zeros_like(basis)
    basis[positive, 0] = 1
    basis[positive, 1] = x
    derivative[positive, 1] = 1
    if knots.size > 2:
        clipped, nonlinear, slopes, _ = _spline_terms(x, np.broadcast_to(knots, (len(x), len(knots))))
        basis[positive, 2:] = nonlinear + slopes * (x - clipped)[:, None]
        derivative[positive, 2:] = slopes
    return basis, derivative


def _expand_basis(basis, designs):
    return np.column_stack((basis, *(design * basis[:, index, None] for index, design in designs.items())))


class _TimeDesign(NamedTuple):
    basis: NDArray[np.float64]
    at_zero: NDArray[np.bool_]
    at_infinity: NDArray[np.bool_]

    def predictor(self, coefficients):
        eta = self.basis @ coefficients
        eta[self.at_zero], eta[self.at_infinity] = -np.inf, np.inf
        return eta


def _observed_information(loss, parameters, parameter_scale, total):
    steps = np.cbrt(np.finfo(float).eps) * np.maximum(1, np.abs(parameters))
    perturbations = np.diag(steps)
    columns = [(loss(parameters + delta)[1] - loss(parameters - delta)[1]) / (2 * step)
        for delta, step in zip(perturbations, steps, strict=True)]
    information = np.column_stack(columns)
    information = (information + information.T) / 2
    if not np.isfinite(information).all():
        return None
    try:
        inverse = cho_solve(cho_factor(information), np.eye(len(parameters)))
    except np.linalg.LinAlgError:
        return None
    return inverse / total / parameter_scale[:, None] / parameter_scale[None, :]


def fit_royston_parmar(
    left: ArrayLike, right: ArrayLike, design: ArrayLike | None = None, *, knots: ArrayLike = _DEFAULT_KNOTS,
    scale: RoystonParmarScale = "hazard", timescale: TimeScale = "log", entry: ArrayLike | None = None,
    weights: ArrayLike | None = None, ancillary: Mapping[int, ArrayLike] | None = None,
    initial: ArrayLike | None = None,
) -> RoystonParmarFit:
    """Fit an RP regression with common knots supplied in the selected time units.

    The columns of design act on gamma0. Ancillary designs are keyed by
    gamma index, starting at 1, and can model time-varying covariate effects.
    The initial vector follows baseline, location, then ascending ancillary
    index order. Design matrices exclude the intercept.
    """
    intervals = _survival_intervals(left, right, entry, weights)
    if not np.isfinite(intervals.right).any():
        raise ValueError("fitting requires at least one observed event or finite censoring interval")
    knots = _real_array(knots, "knots")
    if knots.ndim != 1:
        raise ValueError("fitting requires one vector of common knots")
    _broadcast_inputs(np.array(0.0), np.zeros_like(knots), knots, scale, timescale)
    designs = {0: _design_matrix(design, intervals.keep, "design")}
    if ancillary is not None:
        if not isinstance(ancillary, Mapping) or not all(
            isinstance(j, Integral) and not isinstance(j, bool) and 1 <= j < knots.size for j in ancillary
        ):
            raise ValueError("ancillary must map gamma indices from 1 to len(knots) - 1 to designs")
        designs.update({j: _design_matrix(ancillary[j], intervals.keep, f"ancillary[{j}]") for j in sorted(ancillary)})
    body = []
    with np.errstate(over="ignore", invalid="ignore"):
        for values in (intervals.left, intervals.right, intervals.entry):
            basis, derivative = _time_basis(values, knots, timescale)
            body.append(_TimeDesign(_expand_basis(basis, designs), values == 0, np.isinf(values)))
        _, derivative = _time_basis(intervals.left, knots, timescale)
        derivative = _expand_basis(derivative, designs)
        parameter_scale = np.max([np.max(np.abs(t.basis), axis=0) for t in body], axis=0)
        total = float(intervals.weights.sum())
    if not np.isfinite(parameter_scale).all() or np.any(parameter_scale == 0) or not np.isfinite(total):
        raise ValueError("fit requires an estimable finite design and total weight")
    relative_weights = intervals.weights / total
    low, high, at_entry = (_TimeDesign(t.basis / parameter_scale, t.at_zero, t.at_infinity) for t in body)
    derivative /= parameter_scale
    log_jacobian = _jacobian(intervals, timescale)

    def loss(beta):
        values, l, r, e, d = _log_likelihood_terms(low.predictor(beta), high.predictor(beta),
            at_entry.predictor(beta), derivative @ beta, intervals.exact, log_jacobian, scale)
        if not np.isfinite(values).all() or not all(np.isfinite(a).all() for a in (l, r, e, d)):
            return np.inf, np.full_like(beta, np.nan)
        gradient = (low.basis.T @ (relative_weights * l) + high.basis.T @ (relative_weights * r)
            + at_entry.basis.T @ (relative_weights * e) + derivative.T @ (relative_weights * d))
        return -float(relative_weights @ values), -gradient

    if initial is None:
        observed = np.where(np.isfinite(intervals.right), intervals.left / 2 + intervals.right / 2, intervals.left)
        observed = observed[observed > 0]
        transformed = np.log(observed) if timescale == "log" else observed
        center = np.median(transformed)
        spread = np.std(transformed) or 1
        intercept = np.log(_LOG_TWO) if scale == "hazard" else 0.0
        starts = []
        for size in (1.0, 0.5, 2.0):
            start = np.zeros_like(parameter_scale)
            start[1] = size / spread
            start[0] = intercept - center * start[1]
            starts.append(start)
    else:
        start = _real_array(initial, "initial")
        if start.shape != parameter_scale.shape or not np.isfinite(start).all():
            raise ValueError("initial must be a finite vector with one element per fitted coefficient")
        starts = [start]

    best = None
    for start in starts:
        result = minimize(loss, start * parameter_scale, jac=True, method="BFGS",
            options={"gtol": 1e-9, "maxiter": 1000})
        if np.isfinite(result.fun) and np.max(np.abs(result.jac)) < 2e-7:
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise ValueError("Royston–Parmar likelihood optimization did not converge")

    covariance = _observed_information(loss, best.x, parameter_scale, total)
    parameters = best.x / parameter_scale
    location_stop = knots.size + designs[0].shape[1]
    ancillary_coefficients = {}
    start = location_stop
    for j, design in designs.items():
        if j:
            stop = start + design.shape[1]
            ancillary_coefficients[j] = parameters[start:stop]
            start = stop
    return RoystonParmarFit(parameters[:knots.size], parameters[knots.size:location_stop], ancillary_coefficients,
        covariance, knots.copy(), scale, timescale, -total * float(best.fun), intervals.left.size)
