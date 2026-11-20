"""Parametric ROC smoothing with separately fitted class distributions."""

from dataclasses import dataclass
from math import log, pi, sqrt
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .roc import _binary_inputs
from .roc_bootstrap import Focus, RocDirection, _AreaSpec
from .roc_smoothing import _auc
from .roc_thresholds import _resolve_direction

ParametricDistribution = Literal["normal", "lognormal", "log-normal", "exponential"]
_LOG_SQRT_TWO_PI = log(2 * pi) / 2


@dataclass(frozen=True)
class ParametricDensityFit:
    """Maximum-likelihood fit for one class."""

    distribution: ParametricDistribution
    parameter_names: tuple[str, ...]
    estimates: tuple[float, ...]
    standard_errors: tuple[float, ...]
    covariance: NDArray[np.float64]
    log_likelihood: float
    n_observations: int


@dataclass(frozen=True)
class ParametricROCResult:
    """Smoothed ROC rates, area, and the fitted distribution for each class."""

    specificity: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    auc: float
    direction: RocDirection
    percent: bool
    control_fit: ParametricDensityFit
    case_fit: ParametricDensityFit


def _normal_log_density(x, location, spread):
    with np.errstate(over="ignore"):
        standardized = (x - location) / spread
        return -0.5 * standardized**2 - log(spread) - _LOG_SQRT_TWO_PI


def _log_density(x, distribution, estimates):
    if distribution == "normal":
        return _normal_log_density(x, *estimates)
    result = np.full_like(x, -np.inf)
    if distribution in ("lognormal", "log-normal"):
        support = x > 0
        log_x = np.log(x[support])
        result[support] = _normal_log_density(log_x, *estimates) - log_x
    else:
        support = x >= 0
        rate = estimates[0]
        with np.errstate(over="ignore"):
            result[support] = log(rate) - rate * x[support]
    return result


def _fit_distribution(values, distribution):
    if distribution not in ("normal", "lognormal", "log-normal", "exponential"):
        raise ValueError("distribution must be 'normal', 'lognormal', 'log-normal', or 'exponential'")
    count = len(values)
    if distribution == "exponential":
        if np.any(values < 0) or not np.any(values > 0):
            raise ValueError("exponential fitting requires nonnegative scores with a positive mean")
        maximum = float(values.max())
        with np.errstate(over="ignore", divide="ignore"):
            rate = float(np.divide(1, np.mean(values / maximum) * maximum))
        if not np.isfinite(rate):
            raise ValueError("exponential fitting requires a finite rate")
        names, estimates, errors = ("rate",), (rate,), (rate / sqrt(count),)
    else:
        if distribution in ("lognormal", "log-normal"):
            if np.any(values <= 0):
                raise ValueError("lognormal fitting requires strictly positive scores")
            sample, names = np.log(values), ("meanlog", "sdlog")
        else:
            sample, names = values, ("mean", "sd")
        with np.errstate(over="ignore", invalid="ignore"):
            location = float(np.mean(sample))
            spread = float(np.std(sample, ddof=0))
        if (count < 2 or np.all(sample == sample[0])
            or not np.isfinite(location) or not np.isfinite(spread) or spread <= 0):
            raise ValueError(f"{distribution} fitting requires finite, positive variation")
        estimates = (location, spread)
        errors = (spread / sqrt(count), spread / sqrt(2 * count))
    with np.errstate(over="ignore"):
        covariance = np.diag(np.square(errors))
        log_likelihood = float(_log_density(values, distribution, estimates).sum())
    return ParametricDensityFit(distribution, names, estimates, errors, covariance, log_likelihood, count)


def _cumulative_density(grid, fit, *, reverse=False):
    log_mass = _log_density(grid, fit.distribution, fit.estimates)
    peak = np.max(log_mass)
    if not np.isfinite(peak):
        raise ValueError("score grid must have positive density for both fitted distributions")
    mass = np.exp(log_mass - peak)
    total = np.r_[0.0, np.cumsum(mass[::-1] if reverse else mass)]
    return total / total[-1]


def _rates(grid, control, case, direction):
    # Equal-valued grid entries all share one threshold comparison.
    _, starts, counts = np.unique(grid, return_index=True, return_counts=True)
    if direction == "<":
        specificity = _cumulative_density(grid, control)[starts]
        sensitivity = _cumulative_density(grid, case, reverse=True)[grid.size - starts]
    else:
        ends = starts + counts
        specificity = _cumulative_density(grid, control, reverse=True)[grid.size - ends]
        sensitivity = _cumulative_density(grid, case)[ends]
    specificity, sensitivity = np.repeat(specificity, counts), np.repeat(sensitivity, counts)
    if np.any(np.diff(specificity) < 0):
        return specificity[::-1], sensitivity[::-1]
    return specificity, sensitivity


def smooth_parametric_roc(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    control_distribution: ParametricDistribution = "normal",
    case_distribution: ParametricDistribution = "normal",
    direction: RocDirection = "<",
    n: int = 512,
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
) -> ParametricROCResult:
    """Fit a distribution per class and integrate the pROC fitdistr-style grid.

    Labels use 0 for controls and 1 for cases. Rates use n fitted values on
    the pooled score range plus the two ROC anchors. Normal, lognormal,
    and exponential families are supported, independently for each class.
    """
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    labels, scores = _binary_inputs(labels, scores)
    controls, cases = labels == 0, labels == 1
    if not controls.any() or not cases.any():
        raise ValueError("ROC requires at least one case and one control")
    scores = scores.astype(float, copy=False)
    direction = _resolve_direction(cases, scores, direction)
    area = _AreaSpec.create(partial_auc, focus, standardize, percent)
    control_fit = _fit_distribution(scores[controls], control_distribution)
    case_fit = _fit_distribution(scores[cases], case_distribution)
    start, stop = float(scores.min()), float(scores.max())
    if not np.isfinite(stop - start):
        raise ValueError("score range must be finite")
    grid = np.linspace(start, stop, int(n))
    sp, se = _rates(grid, control_fit, case_fit, direction)
    sp, se = np.r_[0.0, sp, 1.0], np.r_[1.0, se, 0.0]
    auc = _auc(sp, se, None, area, allow_below_chance)
    return ParametricROCResult(sp * area.scale, se * area.scale, auc, direction, percent, control_fit, case_fit)
