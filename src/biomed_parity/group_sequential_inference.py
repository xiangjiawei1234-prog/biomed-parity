"""Repeated and sequential inference for canonical group-sequential analyses."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from .group_sequential import _increment_sd, _next_stage, _spending


@dataclass(frozen=True)
class RepeatedPValues:
    """Stage-specific repeated p-values and their running minima."""

    repeated: NDArray[np.float64]
    cumulative: NDArray[np.float64]

    @property
    def sequential(self) -> float:
        return float(self.cumulative[-1])


@dataclass(frozen=True)
class RepeatedConfidenceIntervals:
    """Confidence interval endpoints at each observed look."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


def _numeric_vector(values, name, *, allow_infinite=False):
    vector = np.asarray(values)
    if vector.ndim != 1 or not vector.size or vector.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty numeric vector")
    if np.isnan(vector).any() or (not allow_infinite and not np.isfinite(vector).all()):
        raise ValueError(f"{name} must contain finite numbers" if not allow_infinite else f"{name} must not contain NaN")
    return vector.astype(float)


def _increasing(values, name):
    vector = _numeric_vector(values, name)
    if vector[0] <= 0 or np.any(np.diff(vector) <= 0):
        raise ValueError(f"{name} must be positive and strictly increasing")
    return vector


def _monitoring_inputs(information, z, planned_final_information, spending_times, parameter, search_interval):
    information = _increasing(information, "information")
    z = _numeric_vector(z, "z")
    if spending_times is None:
        if (
            not isinstance(planned_final_information, Real)
            or isinstance(planned_final_information, (bool, np.bool_))
            or not np.isfinite(planned_final_information) or planned_final_information <= 0
        ):
            raise ValueError("planned_final_information must be positive and finite")
        times = information / planned_final_information
    else:
        times = _increasing(spending_times, "spending_times")
    if len(z) != len(information) or len(times) != len(information):
        raise ValueError("information, z and spending_times must have the same length")
    if np.count_nonzero(times >= 1) > 1:
        raise ValueError("at most the last spending time may reach or exceed 1")
    if parameter is not None and (not isinstance(parameter, Real) or not np.isfinite(parameter)):
        raise ValueError("parameter must be a finite real number")
    interval = _numeric_vector(search_interval, "search_interval")
    if len(interval) != 2 or not 0 < interval[0] < interval[1] < 1:
        raise ValueError("search_interval must be two increasing probabilities strictly between 0 and 1")
    return information / information[-1], z, np.minimum(times, 1), interval


def _calibrate_upper(information, cumulative):
    increments = np.diff(cumulative, prepend=0)
    active = np.flatnonzero(increments > 0)
    bounds = np.full(len(information), np.inf)
    previous = 0.0
    nodes, mass = np.zeros(1), np.ones(1)
    for position, index in enumerate(active):
        time = information[index]
        stage = _next_stage(time, previous, nodes, mass)
        bound = stage.quantile(increments[index], upper=True)
        bounds[index] = bound
        if position + 1 < len(active):
            outgoing_sd = _increment_sd(time, information[active[position + 1]])
            nodes, mass = stage.restrict(-np.inf, bound, outgoing_sd, 1 - cumulative[index])
            previous = time
    return bounds


def _invert_bounds(margin, interval):
    lower, upper = interval
    if margin(lower) <= 0:
        return float(lower)
    if margin(upper) >= 0:
        return float(upper)
    # The normal-quantile scale preserves precision even for tiny p-values.
    root = brentq(lambda quantile: margin(ndtr(quantile)), ndtri(lower), ndtri(upper), xtol=1e-11)
    return float(ndtr(root))


def sequential_pvalue(
    information: ArrayLike,
    z: ArrayLike,
    *,
    planned_final_information: float = 1.0,
    spending_times: ArrayLike | None = None,
    spending: str = "hsd",
    parameter: float | None = None,
    search_interval: ArrayLike = (1e-5, 0.9999),
) -> float:
    """Return the smallest one-sided alpha that would reject by an observed look.

    Information can be a fraction of planned final information, or an
    absolute amount with planned_final_information supplied. Explicit
    spending_times govern spending while information governs correlation.
    This calculation uses one-sided efficacy boundaries, with any
    nonbinding lower stop omitted. Values outside the search interval are
    clipped as in gsDesign::sequentialPValue.
    """
    information, z, times, interval = _monitoring_inputs(
        information, z, planned_final_information, spending_times, parameter, search_interval
    )

    def margin(alpha):
        bounds = _calibrate_upper(information, _spending(times, alpha, spending, parameter))
        return float(np.min(bounds - z))

    return _invert_bounds(margin, interval)


def repeated_pvalues(
    information: ArrayLike,
    z: ArrayLike,
    *,
    planned_final_information: float = 1.0,
    spending_times: ArrayLike | None = None,
    spending: str = "hsd",
    parameter: float | None = None,
    search_interval: ArrayLike = (1e-5, 0.9999),
) -> RepeatedPValues:
    """Invert each observed look's efficacy threshold and return running minima."""
    information, z, times, interval = _monitoring_inputs(
        information, z, planned_final_information, spending_times, parameter, search_interval
    )

    def at_look(index):
        end = index + 1

        def margin(alpha):
            bounds = _calibrate_upper(information[:end], _spending(times[:end], alpha, spending, parameter))
            return float(bounds[-1] - z[index])

        return _invert_bounds(margin, interval)

    values = np.array([at_look(index) for index in range(len(z))])
    return RepeatedPValues(values, np.minimum.accumulate(values))


def repeated_confidence_intervals(
    estimates: ArrayLike,
    standard_errors: ArrayLike,
    upper_z_boundaries: ArrayLike,
    *,
    alternative: str = "greater",
) -> RepeatedConfidenceIntervals:
    """Invert canonical normal efficacy bounds at the observed analyses.

    Supply the critical values applicable to those analyses. 'greater'
    gives lower confidence bounds, 'less' gives upper confidence bounds
    for the reflected efficacy test and 'two-sided' uses both endpoints.
    """
    estimates = _numeric_vector(estimates, "estimates")
    errors = _numeric_vector(standard_errors, "standard_errors")
    bounds = _numeric_vector(upper_z_boundaries, "upper_z_boundaries", allow_infinite=True)
    if len(estimates) != len(errors) or len(estimates) != len(bounds):
        raise ValueError("estimates, standard_errors and upper_z_boundaries must have the same length")
    if np.any(errors <= 0):
        raise ValueError("standard_errors must be positive")
    if np.any(np.isneginf(bounds)):
        raise ValueError("upper_z_boundaries must exceed negative infinity")
    width = errors * bounds
    if alternative == "greater":
        return RepeatedConfidenceIntervals(estimates - width, np.full(len(estimates), np.inf))
    if alternative == "less":
        return RepeatedConfidenceIntervals(np.full(len(estimates), -np.inf), estimates + width)
    if alternative == "two-sided":
        if np.any(bounds < 0):
            raise ValueError("two-sided intervals require nonnegative upper_z_boundaries")
        return RepeatedConfidenceIntervals(estimates - width, estimates + width)
    raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
