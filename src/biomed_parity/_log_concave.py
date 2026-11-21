"""Univariate log-concave maximum likelihood on the observed support."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solveh_banded
from scipy.special import erfcx, exprel, hyp1f1, ndtr

_SQRT_TWO = np.sqrt(2)
_SQRT_TWO_PI = np.sqrt(2 * np.pi)


def _segment_moments(widths, heights):
    change = np.diff(heights)
    # Integrate from the higher endpoint so the relative exponential stays bounded.
    log_scale = np.log(widths) + np.maximum(heights[:-1], heights[1:])
    scale = np.exp(log_scale)
    relative = -np.abs(change)
    return (scale * exprel(relative), scale * hyp1f1(2, 3, relative) / 2,
        scale * hyp1f1(3, 4, relative) / 3)


def _likelihood(widths, weights, heights):
    mass, first, second = _segment_moments(widths, heights)
    increasing = np.diff(heights) > 0
    left = np.where(increasing, first, mass - first)
    right = mass - left
    gradient = np.r_[left, 0] + np.r_[0, right] - weights
    high = mass - 2 * first + second
    left_squared = np.where(increasing, second, high)
    right_squared = np.where(increasing, high, second)
    diagonal = np.r_[left_squared, 0] + np.r_[0, right_squared]
    hessian = np.vstack((diagonal, np.r_[first - second, 0]))
    return float(mass.sum() - weights @ heights), gradient, hessian


def _maximize_on_knots(x, weights, heights):
    widths = np.diff(x)
    heights = heights - np.log(_segment_moments(widths, heights)[0].sum())
    for _ in range(100):
        objective, gradient, hessian = _likelihood(widths, weights, heights)
        hessian[0] += hessian[0].mean() * 1e-12
        step = solveh_banded(hessian, gradient, lower=True)
        decrease = float(gradient @ step)
        # Use the reference's Newton decrement to stop the fit on this knot set.
        if decrease < 1e-7:
            return heights
        size = 1.0
        for _ in range(21):
            trial = heights - size * step
            with np.errstate(over="ignore", invalid="ignore"):
                mass = _segment_moments(widths, trial)[0].sum()
                value = float(mass - weights @ trial)
            if value <= objective:
                # The actual decrease estimates the curvature on the Newton ray.
                ratio = (objective - value) / (size * decrease)
                if ratio < 0.5:
                    trial = heights - (size / (2 * (1 - ratio))) * step
                    mass = _segment_moments(widths, trial)[0].sum()
                heights = trial - np.log(mass)
                break
            size /= 2
        else:
            break
    raise ValueError("log-concave density likelihood failed to converge")


def _interpolate_weights(x, weights, knots):
    index = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    fraction = (x - knots[index]) / (knots[index + 1] - knots[index])
    left = np.bincount(index, weights * (1 - fraction), minlength=knots.size)
    right = np.bincount(index + 1, weights * fraction, minlength=knots.size)
    return left + right


def _convexity(x, heights):
    fraction = (x[1:-1] - x[:-2]) / (x[2:] - x[:-2])
    return heights[:-2] + fraction * (heights[2:] - heights[:-2]) - heights[1:-1]


def _inactive_gradient(x, knots, gradient):
    result = np.zeros_like(gradient)
    for start, stop in zip(knots[:-1], knots[1:]):
        if stop - start <= 1:
            continue
        interior = slice(start + 1, stop)
        left = x[interior] - x[start]
        right = x[stop] - x[interior]
        g = gradient[interior]
        # Each candidate increases a piecewise linear hat between its neighboring knots.
        hat = np.cumsum(g * left) / left + np.cumsum((g * right)[::-1])[::-1] / right - g
        result[interior] = hat * (left * right / (left + right))
    return result


def _log_heights(x, weights):
    knots = np.array([0, x.size - 1])
    feasible = np.zeros(2)
    previous_knots = np.empty(0, dtype=int)
    for _ in range(max(100, 4 * x.size)):
        locations = x[knots]
        target = _maximize_on_knots(locations, _interpolate_weights(x, weights, locations), feasible)
        change = _convexity(locations, target)
        if np.any(change > 1e-11):
            bad = change > 1e-11
            distance = np.maximum(-_convexity(locations, feasible)[bad], 0)
            fractions = distance / (distance + change[bad])
            first = fractions.argmin()
            feasible = feasible + fractions[first] * (target - feasible)
            remove = np.flatnonzero(bad)[first] + 1
            knots, feasible = np.delete(knots, remove), np.delete(feasible, remove)
            continue

        heights = np.interp(x, locations, target)
        _, gradient, _ = _likelihood(np.diff(x), weights, heights)
        derivative = _inactive_gradient(x, knots, gradient)
        new_knot = derivative.argmin()
        if derivative[new_knot] >= -1e-13 or np.array_equal(knots, previous_knots):
            return heights
        previous_knots = knots
        knots = np.sort(np.r_[knots, new_knot])
        feasible = heights[knots]
    raise ValueError("log-concave density active set failed to converge")


def _quantile_fraction(probability, change):
    near_zero = np.abs(change) <= 1e-6
    moderate = ~near_zero & (np.abs(change) < 35)
    large = ~near_zero & ~moderate & np.isfinite(change)
    fraction = probability.copy()
    p, a = probability[near_zero], change[near_zero]
    fraction[near_zero] = p + a * p * (1 - p) / 2
    fraction[moderate] = np.log1p(probability[moderate] * np.expm1(change[moderate])) / change[moderate]
    p, a = probability[large], change[large]
    with np.errstate(divide="ignore"):
        fraction[large] = np.logaddexp(np.log1p(-p), np.log(p) + a) / a
    fraction[np.isposinf(change)] = probability[np.isposinf(change)] > 0
    fraction[np.isneginf(change)] = probability[np.isneginf(change)] == 1
    return np.clip(fraction, 0, 1)


@dataclass(frozen=True)
class _LogConcaveFit:
    origin: float
    span: float
    x: np.ndarray
    heights: np.ndarray
    cumulative: np.ndarray
    bandwidth: float

    def reference_quantile(self, probability):
        index = np.minimum(np.searchsorted(self.cumulative, probability, side="right") - 1,
            self.x.size - 2)
        width = np.diff(self.x)[index]
        fraction = (probability - self.cumulative[index]) / np.diff(self.cumulative)[index]
        # logcondens passes the log-height difference multiplied by the original interval width.
        with np.errstate(over="ignore"):
            change = np.diff(self.heights)[index] * (width * self.span)
        return self.origin + self.span * (self.x[index] + width * _quantile_fraction(fraction, change))

    def cdf(self, scores):
        values = self._scale_scores(scores)
        result = np.where(values >= 1, 1.0, 0.0)
        inside = (values > 0) & (values < 1)
        index = np.searchsorted(self.x, values[inside], side="right") - 1
        fraction = (values[inside] - self.x[index]) / np.diff(self.x)[index]
        change = np.diff(self.heights)[index]
        # logcondens's small-change expansion makes the CDF linear within these segments.
        increasing = change > 0.005
        falling = change < -0.005
        a, u = change[increasing], fraction[increasing]
        fraction[increasing] = np.exp((u - 1) * a) * np.expm1(-u * a) / np.expm1(-a)
        a, u = change[falling], fraction[falling]
        fraction[falling] = np.expm1(u * a) / np.expm1(a)
        result[inside] = self.cumulative[index] + np.diff(self.cumulative)[index] * fraction
        return result

    def _scale_scores(self, scores):
        with np.errstate(over="ignore"):
            return (scores - self.origin) / self.span

    def smooth_cdf(self, scores):
        return np.array([self._smooth_at(value) for value in self._scale_scores(scores)])

    def _smooth_at(self, value):
        sigma = self.bandwidth
        if value <= -40 * sigma:
            return 0.0
        if value >= 1 + 40 * sigma:
            return 1.0

        left, right = self.x[:-1], self.x[1:]
        h_left, h_right = self.heights[:-1], self.heights[1:]
        slope = np.diff(self.heights) / (right - left)
        a, b = (value - left) / sigma, (value - right) / sigma
        fa, fb = ndtr(a), ndtr(b)
        pa, pb = np.exp(-0.5 * a**2), np.exp(-0.5 * b**2)
        area = np.empty_like(slope)
        flat = np.abs(slope) <= 1e-6 * self.span
        area[flat] = np.exp(h_left[flat]) * sigma * (
            a[flat] * fa[flat] - b[flat] * fb[flat] + (pa[flat] - pb[flat]) / _SQRT_TWO_PI
        )
        curved = ~flat
        s = slope[curved] * sigma
        lo, hi = -a[curved] - s, -b[curved] - s
        start_height, end_height = h_left[curved], h_right[curved]
        q = np.empty_like(lo)
        positive = lo >= 0
        negative = hi <= 0
        central = ~positive & ~negative
        # For one-sided intervals, write the normal tails with erfcx so the
        # quadratic exponential cancels before taking their difference.
        left_tail = np.exp(start_height) * pa[curved] / 2
        right_tail = np.exp(end_height) * pb[curved] / 2
        q[positive] = (left_tail[positive] * erfcx(lo[positive] / _SQRT_TWO)
            - right_tail[positive] * erfcx(hi[positive] / _SQRT_TWO))
        q[negative] = (right_tail[negative] * erfcx(-hi[negative] / _SQRT_TWO)
            - left_tail[negative] * erfcx(-lo[negative] / _SQRT_TWO))
        q[central] = np.exp(start_height[central] + s[central] * (a[curved][central] + s[central] / 2)) * (
            ndtr(hi[central]) - ndtr(lo[central])
        )
        area[curved] = (np.exp(end_height) * fb[curved] - np.exp(start_height) * fa[curved] + q) / slope[curved]
        return float(np.clip(area.sum(), 0, 1))


def _fit_log_concave(scores):
    values, counts = np.unique(scores, return_counts=True)
    if values.size < 2:
        raise ValueError("log-concave smoothing requires two distinct scores in each class")
    origin, span = float(values[0]), float(values[-1]) - float(values[0])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("log-concave smoothing requires a finite class score range")
    x = (values - origin) / span
    widths = np.diff(x)
    if np.any(widths <= 0):
        raise ValueError("log-concave smoothing requires distinguishable standardized scores")
    weights = counts / scores.size
    heights = _log_heights(x, weights)
    heights -= np.log(_segment_moments(widths, heights)[0].sum())
    mass, first, second = _segment_moments(widths, heights)
    cumulative = np.r_[0, np.minimum(np.cumsum(mass[:-1]), 1), 1]
    mean = weights @ x
    # The reference applies its finite-sample correction after grouping ties.
    sample_variance = (weights @ (x - mean)**2) * x.size / (x.size - 1)
    increasing = np.diff(heights) > 0
    peak = np.where(increasing, x[1:], x[:-1]) - mean
    step = np.where(increasing, -widths, widths)
    fitted_variance = np.sum(peak**2 * mass + 2 * peak * step * first + step**2 * second)
    if fitted_variance >= sample_variance:
        raise ValueError("log-concave smoothing requires positive convolution variance")
    bandwidth = float(np.sqrt(sample_variance - fitted_variance))
    return _LogConcaveFit(origin, span, x, heights, cumulative, bandwidth)


def log_concave_rates(controls, cases, n, *, smooth):
    control = _fit_log_concave(controls)
    case = _fit_log_concave(cases)
    specificity = np.linspace(0, 1, n)
    thresholds = control.reference_quantile(specificity[1:-1])
    probability = case.smooth_cdf(thresholds) if smooth else case.cdf(thresholds)
    return specificity, np.r_[1, 1 - probability, 0]
