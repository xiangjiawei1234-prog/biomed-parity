"""Turnbull nonparametric likelihood for interval-censored event times."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, minimize


@dataclass(frozen=True)
class TurnbullFit:
    """Probability masses on maximal intersections of observation intervals."""

    intervals: NDArray[np.float64]
    closed: NDArray[np.bool_]
    mass: NDArray[np.float64]
    log_likelihood: float
    optimality_gap: float

    def survival_bounds(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return lower and upper bounds for P(T > t) with these fitted masses."""
        t = np.asarray(times)
        if t.dtype.kind not in "biuf" or np.isnan(t).any():
            raise ValueError("times must be real and cannot be NaN")
        left, right = self.intervals.T
        cumulative = np.r_[np.cumsum(self.mass[::-1])[::-1], 0]
        first_left = np.searchsorted(left, t, side="left")
        index = np.minimum(first_left, len(left) - 1)
        included = (first_left < len(left)) & (left[index] == t) & self.closed[index, 0]
        first_right = np.searchsorted(right, t, side="right")
        return np.stack((cumulative[first_left + included], cumulative[first_right]), axis=-1)


def _observation_intervals(left, right, bounds, weights):
    left, right = np.asarray(left), np.asarray(right)
    if (
        left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf"
        or left.ndim != 1 or not left.size or right.shape != left.shape
        or not np.isfinite(left).all() or np.isnan(right).any()
        or np.any(left < 0) or np.any(right < left)
    ):
        raise ValueError("intervals require 0 <= left <= right, with only right allowed to be infinity")
    closed = np.asarray(bounds)
    if closed.shape != (2,) or closed.dtype.kind not in "biuf" or np.any((closed != 0) & (closed != 1)):
        raise ValueError("bounds must be two booleans indicating which endpoints are closed")
    weights = np.ones(len(left)) if weights is None else np.asarray(weights)
    if (
        weights.shape != left.shape or weights.dtype.kind not in "biuf"
        or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)
    ):
        raise ValueError("weights must be finite, nonnegative, and have positive total")
    keep = weights > 0
    return left[keep].astype(float), right[keep].astype(float), closed.astype(bool), weights[keep].astype(float)


def _maximal_intersections(left, right, endpoint_closed):
    """Order tied endpoints by interval membership and find consecutive left-right pairs."""
    n = len(left)
    points = np.r_[left, right]
    exact = left == right
    left_closed = endpoint_closed[0] | exact
    right_closed = (endpoint_closed[1] | exact) & np.isfinite(right)
    # At a common time: open right, closed left, closed right, open left.
    priority = np.r_[np.where(left_closed, 1, 3), np.where(right_closed, 2, 0)]
    order = np.lexsort((priority, points))
    starts = (order[:-1] < n) & (order[1:] >= n)
    support = np.flatnonzero(starts)
    lo, hi = order[support], order[support + 1] - n
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(2 * n)
    first = np.searchsorted(support, inverse_order[:n], side="left")
    last = np.searchsorted(support + 1, inverse_order[n:], side="right")
    return (
        np.column_stack((left[lo], right[hi])),
        np.column_stack((left_closed[lo], right_closed[hi])),
        first, last,
    )


def _interval_probabilities(mass, first, stop):
    cumulative = np.r_[0, np.cumsum(mass)]
    probability = cumulative[stop] - cumulative[first]
    # Tail intervals can carry positive mass below the precision of a
    # difference between two nearly equal cumulative probabilities.
    for i in np.flatnonzero(probability < 1e-10):
        probability[i] = mass[first[i]:stop[i]].sum()
    return probability


def _newton_mass(mass, first, stop, probability, sqrt_weights, likelihood, value):
    active = np.flatnonzero(mass > 32 * np.finfo(float).eps)
    pivot = mass.argmax()
    others = active[active != pivot]
    if not len(others):
        return None
    pivot_rows = (first <= pivot) & (pivot < stop)
    rows = (first[:, None] <= others) & (others < stop[:, None])
    contrasts = rows.astype(float) - pivot_rows[:, None]
    scaled = contrasts * (sqrt_weights / probability)[:, None]
    coordinates = np.linalg.lstsq(scaled, sqrt_weights, rcond=1e-12)[0]
    direction = np.zeros_like(mass)
    direction[others] = coordinates
    direction[pivot] = -coordinates.sum()
    decreasing = direction < 0
    if not np.any(decreasing):
        return None
    step = min(1.0, np.min(-mass[decreasing] / direction[decreasing]))
    for _ in range(20):
        updated = np.maximum(mass + step * direction, 0)
        updated /= updated.sum()
        improved, _ = likelihood(updated)
        if improved <= value + 2 * np.finfo(float).eps * max(1, value):
            return updated
        step *= 0.5
    return None


def turnbull(
    left: ArrayLike,
    right: ArrayLike,
    *,
    bounds: tuple[bool, bool] = (False, True),
    weights: ArrayLike | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> TurnbullFit:
    """Fit the interval-censored NPMLE with open or closed observation endpoints.

    Equal finite endpoints represent exact events. A zero lower bound or
    infinite upper bound represents left or right censoring, respectively.
    Zero-weight rows are omitted. The likelihood assigns mass to intervals
    and leaves its placement within each interval unspecified.
    """
    left, right, closed, weights = _observation_intervals(left, right, bounds, weights)
    if not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    with np.errstate(over="ignore"):
        total_weight = float(weights.sum())
    if not np.isfinite(total_weight):
        raise ValueError("total weight exceeds the supported numerical range")
    intervals, closed, first, stop = _maximal_intersections(left, right, closed)
    k = len(intervals)
    relative_weights = weights / total_weight
    if np.any(relative_weights == 0):
        raise ValueError("relative weights exceed the supported numerical range")
    sqrt_weights = np.sqrt(relative_weights)

    def likelihood(mass):
        probability = _interval_probabilities(mass, first, stop)
        if np.any(probability <= 0):
            return np.inf, np.zeros(k)
        ratio = relative_weights / probability
        contribution = np.bincount(first, ratio, minlength=k + 1)
        contribution -= np.bincount(stop, ratio, minlength=k + 1)
        return -float(relative_weights @ np.log(probability)), -np.cumsum(contribution[:-1])

    def objective(mass):
        value, gradient = likelihood(mass)
        return value + (mass.sum() - 1), gradient + 1

    if np.all(stop == first + 1):
        mass = np.bincount(first, relative_weights, minlength=k)
        remaining = 0
    else:
        start = np.full(k, 1 / k)
        result = minimize(
            objective, start, jac=True, method="SLSQP", bounds=[(0, 1)] * k,
            constraints={"type": "eq", "fun": lambda p: p.sum() - 1, "jac": lambda p: np.ones(k)},
            options={"maxiter": max_iterations, "ftol": min(1e-13, tolerance * 1e-3)},
        )
        mass = np.maximum(result.x, 0)
        remaining = max_iterations - result.nit
    mass /= mass.sum()
    value, gradient = likelihood(mass)
    for _ in range(remaining):
        best = int(gradient.argmin())
        if not np.isfinite(value) or -gradient[best] - 1 <= tolerance:
            break
        probability = _interval_probabilities(mass, first, stop)
        if mass[best] > 32 * np.finfo(float).eps:
            updated = _newton_mass(mass, first, stop, probability, sqrt_weights, likelihood, value)
            if updated is None:
                mass *= np.maximum(-gradient, 0)
                mass /= mass.sum()
            else:
                mass = updated
        else:
            # The EM update leaves every zero mass unchanged. An improving
            # inactive interval can re-enter along an exact simplex line search.
            direction = ((first <= best) & (best < stop)) - probability

            def line_derivative(step):
                with np.errstate(divide="ignore"):
                    return -relative_weights @ (direction / (probability + step * direction))

            step = brentq(line_derivative, 0, 1)
            mass *= 1 - step
            mass[best] += step
        value, gradient = likelihood(mass)
    gap = max(-float(gradient.min()) - 1, 0.0)
    if not np.isfinite(value) or gap > tolerance:
        raise ValueError("Turnbull likelihood optimization did not attain the requested tolerance")
    return TurnbullFit(intervals, closed, mass, -total_weight * value, gap)
