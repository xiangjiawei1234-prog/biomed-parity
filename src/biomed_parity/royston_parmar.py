"""Royston–Parmar spline survival distributions on supplied coefficients."""

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import erfcx, log_ndtr, ndtri_exp

RoystonParmarScale = Literal["hazard", "odds", "normal"]
TimeScale = Literal["log", "identity"]

_DEFAULT_KNOTS = (-10.0, 10.0)
_LOG_TWO = log(2)
_LOG_SQRT_TWO_PI = log(2 * pi) / 2
_SQRT_TWO = sqrt(2)
_LOG_MILLS_FACTOR = log(2 / pi) / 2


@dataclass(frozen=True)
class RoystonParmarResult:
    """Distribution values and log rates at the requested times."""

    survival: NDArray[np.float64]
    cumulative_hazard: NDArray[np.float64]
    hazard: NDArray[np.float64]
    density: NDArray[np.float64]
    cdf: NDArray[np.float64]
    log_survival: NDArray[np.float64]
    log_hazard: NDArray[np.float64]
    log_density: NDArray[np.float64]


def _real_array(value, name):
    values = np.asarray(value)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    return values.astype(float, copy=False)


def _broadcast_inputs(values, gamma, knots, scale, timescale):
    if scale not in ("hazard", "odds", "normal"):
        raise ValueError("scale must be 'hazard', 'odds', or 'normal'")
    if timescale not in ("log", "identity"):
        raise ValueError("timescale must be 'log' or 'identity'")
    gamma = _real_array(gamma, "gamma")
    knots = _real_array(knots, "knots")
    if not gamma.ndim or not knots.ndim or gamma.shape[-1] != knots.shape[-1] or knots.shape[-1] < 2:
        raise ValueError("gamma and knots must have the same final dimension of at least two")
    if not np.isfinite(gamma).all() or not np.isfinite(knots).all():
        raise ValueError("gamma and knots must be finite")
    if np.any(knots[..., 1:] <= knots[..., :-1]):
        raise ValueError("knots must be strictly increasing, including boundary knots")
    with np.errstate(over="ignore"):
        if not np.isfinite(knots[..., -1] - knots[..., 0]).all():
            raise ValueError("boundary knot spacing must be finite")
    try:
        shape = np.broadcast_shapes(values.shape, gamma.shape[:-1], knots.shape[:-1])
    except ValueError as error:
        raise ValueError("values and the leading dimensions of gamma and knots must broadcast") from error
    parameter_shape = shape + knots.shape[-1:]
    return (np.broadcast_to(values, shape), np.broadcast_to(gamma, parameter_shape),
        np.broadcast_to(knots, parameter_shape))


def _spline_terms(x, knots):
    lower, upper = knots[:, 0], knots[:, -1]
    middle = knots[:, 1:-1]
    left_gap = middle - lower[:, None]
    right_gap = upper[:, None] - middle
    span = upper - lower
    weight = right_gap / span[:, None]

    # Evaluate at the upper boundary before extending by the exact linear tail.
    x_inside = np.minimum(x, upper)
    offset = np.maximum(x_inside - lower, 0)[:, None]
    middle_offset = np.maximum(x_inside[:, None] - middle, 0)
    basis = middle_offset**3 - weight * offset**3
    derivative = 3 * (middle_offset**2 - weight * offset**2)
    tail = x >= upper
    slope_factor = right_gap[tail] * left_gap[tail]
    basis[tail] = -slope_factor * (span[tail, None] + right_gap[tail])
    derivative[tail] = -3 * slope_factor
    return x_inside, basis, derivative, tail


def _spline_at(x, gamma, knots):
    if gamma.shape[-1] == 2:
        return gamma[:, 0] + gamma[:, 1] * x, gamma[:, 1]

    x_inside, basis, derivative, tail = _spline_terms(x, knots)
    slopes = gamma[:, 1] + np.sum(derivative * gamma[:, 2:], axis=1)
    eta = gamma[:, 0] + gamma[:, 1] * x_inside + np.sum(basis * gamma[:, 2:], axis=1)
    eta[tail] += slopes[tail] * (x[tail] - x_inside[tail])
    return eta, slopes


def _link_values(eta, scale):
    if scale == "hazard":
        cumulative = np.exp(eta)
        log_density = np.subtract(eta, cumulative, out=np.full_like(eta, -np.inf), where=np.isfinite(eta))
        return cumulative, eta, log_density
    if scale == "odds":
        cumulative = np.logaddexp(0, eta)
        log_hazard = -np.logaddexp(0, -eta)
        return cumulative, log_hazard, log_hazard - cumulative

    cumulative = -log_ndtr(-eta)
    log_density = -0.5 * eta**2 - _LOG_SQRT_TWO_PI
    log_hazard = log_density.copy()
    upper = eta > 0
    log_hazard[upper] = _LOG_MILLS_FACTOR - np.log(erfcx(eta[upper] / _SQRT_TWO))
    log_hazard[~upper] += cumulative[~upper]
    return cumulative, log_hazard, log_density


def _infinite_log_hazard(gamma, knots, scale, timescale):
    boundary = knots[:, -1]
    eta, slope = _spline_at(boundary, gamma, knots)
    result = np.full_like(slope, -np.inf)
    if timescale == "identity":
        positive = slope > 0
        result[positive] = np.log(slope[positive]) if scale == "odds" else np.inf
    elif scale == "hazard":
        result[slope > 1] = np.inf
        exponential = slope == 1
        result[exponential] = eta[exponential] - boundary[exponential]
    return result


def royston_parmar(
    time: ArrayLike,
    gamma: ArrayLike,
    *,
    knots: ArrayLike = _DEFAULT_KNOTS,
    scale: RoystonParmarScale = "hazard",
    timescale: TimeScale = "log",
) -> RoystonParmarResult:
    """Evaluate the flexsurv RP spline basis and survival distribution.

    Supply boundary and interior knots on the selected time scale. The
    final axis of gamma contains the intercept, slope, and one term per
    interior knot. Leading dimensions broadcast with time. Nonpositive
    times have zero cumulative hazard, density, and instantaneous hazard.
    """
    time, gamma, knots = _broadcast_inputs(_real_array(time, "time"), gamma, knots, scale, timescale)
    cumulative = np.zeros(time.shape)
    log_hazard = np.full(time.shape, -np.inf)
    log_density = np.full(time.shape, -np.inf)
    finite = (time > 0) & np.isfinite(time)
    x = np.log(time[finite]) if timescale == "log" else time[finite]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        eta, slope = _spline_at(x, gamma[finite], knots[finite])
        area, hazard_factor, density_factor = _link_values(eta, scale)
        derivative = np.full_like(slope, -np.inf)
        positive = slope > 0
        derivative[positive] = np.log(slope[positive])
        if timescale == "log":
            derivative[positive] -= x[positive]
        cumulative[finite] = area
        log_hazard[finite] = np.add(derivative, hazard_factor, out=derivative.copy(), where=positive)
        log_density[finite] = derivative + density_factor

        infinite = np.isposinf(time)
        cumulative[infinite] = np.inf
        if infinite.any():
            log_hazard[infinite] = _infinite_log_hazard(gamma[infinite], knots[infinite], scale, timescale)

        missing = np.isnan(time)
        cumulative[missing] = log_hazard[missing] = log_density[missing] = np.nan
        log_survival = -cumulative
        return RoystonParmarResult(np.exp(log_survival), cumulative, np.exp(log_hazard), np.exp(log_density),
            -np.expm1(log_survival), log_survival, log_hazard, log_density)


def _log_complement(log_probability):
    with np.errstate(divide="ignore"):
        lower = log_probability < -_LOG_TWO
        result = np.empty_like(log_probability)
        result[lower] = np.log1p(-np.exp(log_probability[lower]))
        result[~lower] = np.log(-np.expm1(log_probability[~lower]))
        return result


def _quantile_target(probability, scale, lower_tail, log_p):
    invalid = (probability > (0 if log_p else 1)) | ((probability < 0) & (not log_p))
    if np.any(invalid):
        raise ValueError("probability must lie in [0, 1], or be a nonpositive log probability")
    with np.errstate(divide="ignore"):
        direct = probability if log_p else np.log(probability)
        complement = _log_complement(direct) if log_p else np.log1p(-probability)
        log_cdf, log_survival = (direct, complement) if lower_tail else (complement, direct)
        if scale == "odds":
            target = log_cdf - log_survival
        elif scale == "normal":
            target = np.where(log_cdf < -_LOG_TWO, ndtri_exp(log_cdf), -ndtri_exp(log_survival))
        else:
            # At very small F, log(-log(1-F)) equals log(F) to working precision.
            target = np.where(log_cdf < -36, log_cdf, np.log(-log_survival))
    return target


def _monotone_grid(gamma, knots, timescale):
    grid = knots if timescale == "log" else np.r_[0.0, knots[knots > 0]]

    def basis(x):
        return _spline_at(x, np.broadcast_to(gamma, (len(x), len(gamma))),
            np.broadcast_to(knots, (len(x), len(knots))))

    eta, endpoint_slopes = basis(grid)
    if not np.isfinite(eta).all() or not np.isfinite(endpoint_slopes).all():
        raise ValueError("the spline must be finite at the quantile brackets")
    if endpoint_slopes[-1] <= 0:
        raise ValueError("quantiles require a positive right-tail spline slope")
    if timescale == "log" and endpoint_slopes[0] < 0:
        raise ValueError("quantiles require a nondecreasing spline")

    # The derivative on an interval is quadratic. Check its endpoints and any interior minimum.
    left, right = endpoint_slopes[:-1], endpoint_slopes[1:]
    _, middle = basis(grid[:-1] + np.diff(grid) / 2)
    quadratic = 2 * (left + right - 2 * middle)
    linear = right - left - quadratic
    with np.errstate(over="ignore"):
        vertex = np.divide(-linear, 2 * quadratic, out=np.zeros_like(linear), where=quadratic > 0)
    interior = (vertex > 0) & (vertex < 1)
    minimum = np.minimum(left, right)
    minimum[interior] = left[interior] + vertex[interior] * (linear[interior] + vertex[interior] * quadratic[interior])
    tolerance = 32 * np.finfo(float).eps * np.maximum(1, np.maximum(np.abs(left), np.abs(right)))
    if np.any(minimum < -tolerance):
        raise ValueError("quantiles require a nondecreasing spline")
    return grid, eta, endpoint_slopes


def _invert_spline(target, gamma, knots, timescale, grid, eta, slopes):
    if target <= eta[0]:
        if timescale == "identity" or slopes[0] == 0:
            return -np.inf if timescale == "log" else 0.0
        return (target - gamma[0]) / slopes[0]
    if gamma.size == 2:
        return (target - gamma[0]) / gamma[1]
    if target > eta[-1]:
        return grid[-1] + (target - eta[-1]) / slopes[-1]

    index = np.searchsorted(eta, target, side="left")
    if target == eta[index]:
        return grid[index]

    start, stop = grid[index - 1], grid[index]

    def equation(fraction):
        x = np.array([(1 - fraction) * start + fraction * stop])
        observed, _ = _spline_at(x, gamma[None, :], knots[None, :])
        return observed[0] - target

    fraction = brentq(equation, 0.0, 1.0, xtol=1e-14, rtol=1e-14)
    return (1 - fraction) * start + fraction * stop


def royston_parmar_quantile(
    probability: ArrayLike,
    gamma: ArrayLike,
    *,
    knots: ArrayLike = _DEFAULT_KNOTS,
    scale: RoystonParmarScale = "hazard",
    timescale: TimeScale = "log",
    lower_tail: bool = True,
    log_p: bool = False,
) -> NDArray[np.float64]:
    """Invert a nondecreasing RP spline on the predictor scale.

    The supplied probabilities can use either tail and the log scale.
    The endpoint for a zero CDF probability follows flexsurv and is -inf.
    A spline with a positive right-tail slope defines a proper upper tail.
    """
    probability, gamma, knots = _broadcast_inputs(_real_array(probability, "probability"), gamma, knots, scale, timescale)
    target = np.asarray(_quantile_target(probability, scale, lower_tail, log_p))
    result = target.copy()
    grids = {}
    for index in np.ndindex(target.shape):
        if not np.isfinite(target[index]):
            continue
        coefficients, boundaries = gamma[index], knots[index]
        key = (coefficients.tobytes(), boundaries.tobytes())
        if key not in grids:
            grids[key] = _monotone_grid(coefficients, boundaries, timescale)
        with np.errstate(over="ignore"):
            value = _invert_spline(target[index], coefficients, boundaries, timescale, *grids[key])
            result[index] = np.exp(value) if timescale == "log" else value
    return result
