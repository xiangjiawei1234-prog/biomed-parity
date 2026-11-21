"""Semiparametric proportional hazards and survival odds with interval censoring."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from .interval_censoring import (
    IntervalDistribution, _interval_probabilities, _maximal_intersections, _observation_intervals,
)


@dataclass(frozen=True)
class IntervalRegressionFit:
    """Fitted censoring likelihood and distribution at a zero covariate vector."""

    coefficients: NDArray[np.float64]
    baseline: IntervalDistribution
    model: str
    log_likelihood: float
    _reference: NDArray[np.float64]
    _centered_survival: NDArray[np.float64]

    def distribution(self, moderators: ArrayLike) -> IntervalDistribution:
        """Return a fitted interval distribution for one supplied covariate vector."""
        row = _real_design(moderators)
        if row.shape != self.coefficients.shape:
            raise ValueError("moderators must contain one value per fitted coefficient")
        linear = (row - self._reference) @ self.coefficients
        survival = _survival(self._centered_survival, linear, self.model)
        return IntervalDistribution(self.baseline.intervals, self.baseline.closed, -np.diff(survival))


def _real_design(value):
    result = np.asarray(value)
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError("moderators must contain finite real numbers")
    return result.astype(float)


def _survival(baseline, linear, model):
    if model == "ph":
        result = np.empty_like(baseline)
        result[baseline == 0] = 0
        result[baseline == 1] = 1
        inside = (baseline > 0) & (baseline < 1)
        with np.errstate(over="ignore", under="ignore"):
            result[inside] = np.exp(-np.exp(linear + np.log(-np.log(baseline[inside]))))
        return result
    return expit(logit(baseline) + linear)


def _ph_log_probability(left, width, linear):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = np.exp(linear)
        cumulative = left * ratio
        scaled = width * ratio
        factor = -np.expm1(-scaled)
        log_probability = -cumulative + np.log(factor)
        inverse = np.exp(-scaled) / factor
        score_linear = -cumulative + np.where(np.isfinite(scaled), scaled * inverse, 0)
        score_width = ratio * inverse
    return log_probability, score_linear, -ratio, score_width


def _odds_log_probability(left, width, linear):
    # Positive coefficients increase log survival odds, as in icenReg.
    eta = -linear
    right = left + width
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        left_denominator = np.logaddexp(-left, eta + np.log(-np.expm1(-left)))
        right_denominator = np.logaddexp(-right, eta + np.log(-np.expm1(-right)))
        log_probability = eta - left + np.log(-np.expm1(-width)) - left_denominator - right_denominator
        log_survival_left, log_survival_right = -left - left_denominator, -right - right_denominator
        survival_left, survival_right = np.exp(log_survival_left), np.exp(log_survival_right)
        derivative_left = np.exp(eta + log_survival_left) - survival_left
        derivative_right = np.exp(eta + log_survival_right) - survival_right
        score_linear = 1 - survival_left - survival_right
        score_left = -1 - derivative_left - derivative_right
        score_width = np.exp(-width) / -np.expm1(-width) - derivative_right
    return log_probability, score_linear, score_left, score_width


def _loss(parameters, design, weights, first, stop, log_probability):
    n_predictors = design.shape[1]
    hazard = parameters[n_predictors:]
    after = stop == len(hazard) + 1
    past_hazard = np.r_[0, np.cumsum(hazard)][first]
    width = np.full_like(past_hazard, np.inf)
    width[~after] = _interval_probabilities(hazard, first[~after], stop[~after])
    values, linear_score, left_score, width_score = log_probability(
        past_hazard, width, design @ parameters[:n_predictors],
    )
    if not all(np.isfinite(part).all() for part in (values, linear_score, left_score, width_score)):
        return np.inf, np.zeros_like(parameters)
    # A baseline increment contributes to every later left boundary and to
    # every finite observation interval that contains that increment.
    size = len(hazard) + 1
    accumulated = np.zeros(size)
    accumulated[0] = weights @ left_score
    accumulated -= np.bincount(first, weights * left_score, minlength=size)
    local = weights[~after] * width_score[~after]
    accumulated += np.bincount(first[~after], local, minlength=size)
    accumulated -= np.bincount(stop[~after], local, minlength=size)
    score = np.r_[design.T @ (weights * linear_score), np.cumsum(accumulated)[:-1]]
    return -float(weights @ values), -score


def fit_interval_regression(
    left: ArrayLike,
    right: ArrayLike,
    moderators: ArrayLike,
    *,
    model: Literal["ph", "po"] = "ph",
    bounds: tuple[bool, bool] = (False, True),
    weights: ArrayLike | None = None,
) -> IntervalRegressionFit:
    """Estimate coefficients and a nonparametric baseline on Turnbull support."""
    if model not in ("ph", "po"):
        raise ValueError("model must be 'ph' or 'po'")
    left, right, closed, weights, positive = _observation_intervals(left, right, bounds, weights)
    design = _real_design(moderators)
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or len(design) != len(positive) or design.shape[1] == 0:
        raise ValueError("moderators require one design row per observation and at least one column")
    design = design[positive]
    center = design.mean(axis=0)
    design -= center
    scale = np.max(np.abs(design), axis=0)
    if np.any(scale == 0) or np.linalg.matrix_rank(design / scale) != design.shape[1]:
        raise ValueError("moderators must form a full-rank design without an intercept")
    design /= scale

    intervals, closed, first, stop = _maximal_intersections(left, right, closed)
    if len(intervals) < 2:
        raise ValueError("at least two maximal intersections are needed for regression")
    with np.errstate(over="ignore"):
        total = float(weights.sum())
    if not np.isfinite(total):
        raise ValueError("total weight exceeds the supported numerical range")
    weights /= total
    if np.any(weights == 0):
        raise ValueError("relative weights exceed the supported numerical range")
    probability = _ph_log_probability if model == "ph" else _odds_log_probability

    n_predictors = design.shape[1]
    k = len(intervals)
    hazard = -np.diff(np.log(np.arange(k, 0, -1, dtype=float) / k))
    parameter_bounds = [(None, None)] * n_predictors + [(0.0, None)] * (k - 1)
    starts = [np.r_[np.zeros(n_predictors), multiplier * hazard] for multiplier in (1, 0.5, 2)]
    optimizers = (
        ("L-BFGS-B", {"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-9, "maxls": 50}),
        ("SLSQP", {"maxiter": 2000, "ftol": 5e-14}),
    )
    best = None
    for optimizer, options in optimizers:
        for start in starts:
            result = minimize(
                _loss, start, args=(design, weights, first, stop, probability), jac=True,
                method=optimizer, bounds=parameter_bounds, options=options,
            )
            gradient = result.jac.copy()
            inactive = result.x[n_predictors:] == 0
            gradient[n_predictors:][inactive] = np.minimum(gradient[n_predictors:][inactive], 0)
            if np.isfinite(result.fun) and np.max(np.abs(gradient)) < 1e-6:
                if best is None or result.fun < best.fun:
                    best = result
        if best is not None:
            break
    if best is None:
        raise ValueError("interval regression likelihood optimization did not converge")
    coefficients = best.x[:n_predictors] / scale
    centered_survival = np.r_[1, np.exp(-np.cumsum(best.x[n_predictors:])), 0]
    survival = _survival(centered_survival, -center @ coefficients, model)
    baseline = IntervalDistribution(intervals, closed, -np.diff(survival))
    return IntervalRegressionFit(
        coefficients, baseline, model, -total * float(best.fun), center, centered_survival,
    )
