"""Exact group-sequential inference from cumulative binomial counts."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

from .group_sequential import _spending


@dataclass(frozen=True)
class BinomialCrossingProbabilities:
    """First crossings by analysis (rows) and success probability (columns)."""

    sample_sizes: NDArray[np.intp]
    probabilities: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    continuing: NDArray[np.float64]
    expected_sample_size: NDArray[np.float64]

    @property
    def lower_total(self) -> NDArray[np.float64]:
        return self.lower.sum(axis=0)

    @property
    def upper_total(self) -> NDArray[np.float64]:
        return self.upper.sum(axis=0)


@dataclass(frozen=True)
class BinomialHarmProbabilities(BinomialCrossingProbabilities):
    """The upper crossing probability split into exclusive futility and harm."""

    futility: NDArray[np.float64]
    harm: NDArray[np.float64]


@dataclass(frozen=True)
class BinomialEfficacyBoundaries:
    """Integer efficacy thresholds and target and achieved cumulative alpha."""

    sample_sizes: NDArray[np.intp]
    lower_boundaries: NDArray[np.intp]
    upper_boundaries: NDArray[np.intp]
    cumulative_alpha: NDArray[np.float64]
    cumulative_error: NDArray[np.float64]


@dataclass(frozen=True)
class BinomialSPRT:
    """Integer and continuous likelihood-ratio boundaries with exact operating characteristics."""

    lower_boundaries: NDArray[np.intp]
    upper_boundaries: NDArray[np.intp]
    lower_intercept: float
    upper_intercept: float
    slope: float
    operating_characteristics: BinomialCrossingProbabilities


def _integer_vector(values, name):
    vector = np.asarray(values)
    if (
        vector.ndim != 1 or not vector.size or vector.dtype.kind not in "iuf"
        or not np.isfinite(vector).all()
        or (vector.dtype.kind == "f" and np.any(vector != np.floor(vector)))
        or np.any(vector >= np.iinfo(np.intp).max) or np.any(vector < -1)
    ):
        raise ValueError(f"{name} must be a nonempty integer vector with entries at least -1")
    return vector.astype(np.intp)


def _sample_sizes(sample_sizes):
    sizes = _integer_vector(sample_sizes, "sample_sizes")
    if sizes[0] <= 0 or np.any(sizes[1:] <= sizes[:-1]):
        raise ValueError("sample_sizes must be positive and strictly increasing")
    return sizes


def _thresholds(values, name, sizes, *, lower=False):
    vector = _integer_vector(values, name)
    if vector.shape != sizes.shape:
        raise ValueError(f"{name} must contain one threshold per analysis")
    if lower and np.any(vector > sizes):
        raise ValueError(f"{name} must not exceed sample_sizes")
    if not lower and np.any(vector < 0):
        raise ValueError(f"{name} must be nonnegative")
    return vector


def _probabilities(values):
    vector = np.atleast_1d(np.asarray(values))
    if (
        vector.ndim != 1 or not vector.size or vector.dtype.kind not in "iuf"
        or not np.isfinite(vector).all() or np.any((vector < 0) | (vector > 1))
    ):
        raise ValueError("probabilities must contain finite probabilities in [0, 1]")
    return vector.astype(float)


def _check_threshold_order(lower, upper):
    if np.any(lower >= upper):
        raise ValueError("lower_boundaries must be strictly below the active upper stopping thresholds")


def _propagate(sizes, lower, upper, probabilities, *, initial_size=0, initial_count=0, harm=None):
    increment = np.diff(sizes, prepend=initial_size)
    support = [np.arange(count + 1) for count in increment]
    shape = (len(sizes), len(probabilities))
    low_prob, high_prob, cont_prob = np.zeros((3, *shape))
    if harm is not None:
        futility_prob, harm_prob = np.zeros((2, *shape))
    for j, p in enumerate(probabilities):
        remaining = np.ones(1)
        offset = initial_count
        for i, possible in enumerate(support):
            mass = np.convolve(remaining, binom.pmf(possible, increment[i], p))
            left = max(0, lower[i] - offset + 1)
            right = max(0, upper[i] - offset)
            low_prob[i, j] = mass[:left].sum()
            high_prob[i, j] = mass[right:].sum()
            if harm is not None:
                harm_start = max(0, harm[i] - offset)
                futility_prob[i, j] = mass[right:harm_start].sum()
                harm_prob[i, j] = mass[harm_start:].sum()
            remaining = mass[left:right]
            cont_prob[i, j] = remaining.sum()
            if not cont_prob[i, j]:
                break
            offset += left
    # The first reported analysis is reached with probability one. Each
    # later increment is accrued only after continuing at the prior look.
    expected = sizes[0] + np.diff(sizes) @ cont_prob[:-1]
    common = (sizes, probabilities, low_prob, high_prob, cont_prob, expected)
    if harm is None:
        return BinomialCrossingProbabilities(*common)
    return BinomialHarmProbabilities(*common, futility_prob, harm_prob)


def _prepare(sample_sizes, lower_boundaries, upper_boundaries, probabilities, respect_lower):
    sizes = _sample_sizes(sample_sizes)
    lower = _thresholds(lower_boundaries, "lower_boundaries", sizes, lower=True)
    upper = _thresholds(upper_boundaries, "upper_boundaries", sizes)
    _check_threshold_order(lower, upper)
    if not isinstance(respect_lower, (bool, np.bool_)):
        raise ValueError("respect_lower must be boolean")
    if not respect_lower:
        lower.fill(-1)
    return sizes, lower, upper, _probabilities(probabilities)


def binomial_crossing_probabilities(
    sample_sizes: ArrayLike,
    lower_boundaries: ArrayLike,
    upper_boundaries: ArrayLike,
    probabilities: ArrayLike,
    *,
    respect_lower: bool = True,
) -> BinomialCrossingProbabilities:
    """Evaluate exact first crossings of inclusive integer success-count bounds.

    A lower threshold of -1 or an upper threshold of n + 1 disables that
    boundary. Set respect_lower=False to evaluate upper crossings ignoring
    lower stopping. Trials unresolved at the final look stop there.
    """
    inputs = _prepare(sample_sizes, lower_boundaries, upper_boundaries, probabilities, respect_lower)
    return _propagate(*inputs)


def binomial_conditional_probabilities(
    sample_sizes: ArrayLike,
    lower_boundaries: ArrayLike,
    upper_boundaries: ArrayLike,
    probabilities: ArrayLike,
    *,
    look: int,
    successes: int,
    respect_lower: bool = True,
) -> BinomialCrossingProbabilities:
    """Evaluate future crossings given a zero-based interim look and count."""
    sizes, lower, upper, probabilities = _prepare(
        sample_sizes, lower_boundaries, upper_boundaries, probabilities, respect_lower
    )
    if not isinstance(look, Integral) or isinstance(look, bool) or not 0 <= look < len(sizes) - 1:
        raise ValueError("look must be the zero-based index of a nonfinal analysis")
    if not isinstance(successes, Integral) or isinstance(successes, bool) or not 0 <= successes <= sizes[look]:
        raise ValueError("successes must be an integer from zero through the interim sample size")
    future = slice(look + 1, None)
    return _propagate(
        sizes[future], lower[future], upper[future], probabilities,
        initial_size=sizes[look], initial_count=successes,
    )


def _active_flags(flags, name, n):
    flags = np.asarray(flags)
    if flags.dtype.kind != "b" or flags.shape not in ((), (n,)):
        raise ValueError(f"{name} must be a boolean or boolean vector with one entry per analysis")
    return flags


def binomial_harm_probabilities(
    sample_sizes: ArrayLike,
    efficacy_boundaries: ArrayLike,
    futility_boundaries: ArrayLike,
    harm_boundaries: ArrayLike,
    probabilities: ArrayLike,
    *,
    test_futility: ArrayLike = True,
    test_harm: ArrayLike = True,
) -> BinomialHarmProbabilities:
    """Evaluate lower-tail efficacy and upper-tail futility and harm stops.

    When an observed count crosses both upper thresholds it is assigned
    to harm. Setting a flag to False disables that stop at its analysis.
    Use lower and upper for the efficacy and combined upper crossing.
    """
    sizes = _sample_sizes(sample_sizes)
    lower = _thresholds(efficacy_boundaries, "efficacy_boundaries", sizes, lower=True)
    futility = _thresholds(futility_boundaries, "futility_boundaries", sizes)
    harm = _thresholds(harm_boundaries, "harm_boundaries", sizes)
    futility = np.where(_active_flags(test_futility, "test_futility", len(sizes)), futility, sizes + 1)
    harm = np.where(_active_flags(test_harm, "test_harm", len(sizes)), harm, sizes + 1)
    upper = np.minimum(futility, harm)
    _check_threshold_order(lower, upper)
    return _propagate(sizes, lower, upper, _probabilities(probabilities), harm=harm)


def _open_probability(value, name):
    if not isinstance(value, Real) or not 0 < value < 1:
        raise ValueError(f"{name} must be a scalar probability in (0, 1)")
    return float(value)


def binomial_efficacy_boundaries(
    sample_sizes: ArrayLike,
    null_probability: float,
    *,
    alpha: float = 0.025,
    direction: str = "upper",
    spending: str = "hsd",
    parameter: float | None = None,
    spending_times: ArrayLike | None = None,
) -> BinomialEfficacyBoundaries:
    """Calibrate nonrandomized one-sided exact binomial spending boundaries.

    At each analysis, use the largest reachable crossing probability that
    fits the remaining cumulative spend. Early unspent alpha carries over.
    Independent spending times default to sample size / final sample size.
    """
    sizes = _sample_sizes(sample_sizes)
    p = _open_probability(null_probability, "null_probability")
    alpha = _open_probability(alpha, "alpha")
    if direction not in ("upper", "lower"):
        raise ValueError("direction must be 'upper' or 'lower'")
    if parameter is not None and (not isinstance(parameter, Real) or not np.isfinite(parameter)):
        raise ValueError("parameter must be a finite real number")
    if spending_times is None:
        times = sizes / sizes[-1]
    else:
        times = np.asarray(spending_times)
        if (
            times.shape != sizes.shape or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or times[0] <= 0
            or np.any(times[1:] <= times[:-1]) or np.count_nonzero(times >= 1) > 1
        ):
            raise ValueError("spending_times must be positive, increasing and reach 1 only at the last look")
        times = np.minimum(times, 1)
    target = _spending(times, alpha, spending, parameter)
    target[times == 1] = alpha
    reflect = direction == "upper"
    # Allow for floating-point error when a discrete tail equals the spending target.
    spend_limit = target * (1 + 64 * np.finfo(float).eps)

    boundaries = np.empty(len(sizes), dtype=np.intp)
    achieved = np.empty(len(sizes))
    remaining, offset, spent = np.ones(1), 0, 0.0
    for i, n in enumerate(np.diff(sizes, prepend=0)):
        counts = np.arange(n + 1)
        increment_mass = binom.pmf(counts[::-1] if reflect else counts, n, p)
        mass = np.convolve(remaining, increment_mass)
        cumulative = np.cumsum(mass)
        available = max(0, spend_limit[i] - spent)
        # With a budget below one, the extreme count must remain eligible to continue.
        count = min(int(np.searchsorted(cumulative, available, side="right")), mass.size - 1)
        boundaries[i] = offset + count - 1
        if count:
            spent += cumulative[count - 1]
        achieved[i] = spent
        remaining = mass[count:]
        offset += count
    if reflect:
        lower, upper = np.full_like(sizes, -1), sizes - boundaries
    else:
        lower, upper = boundaries, sizes + 1
    return BinomialEfficacyBoundaries(sizes, lower, upper, target, achieved)


def binomial_sprt(
    p0: float = 0.05,
    p1: float = 0.25,
    *,
    alpha: float = 0.1,
    beta: float = 0.15,
    min_sample_size: int = 10,
    max_sample_size: int = 35,
) -> BinomialSPRT:
    """Truncate the binomial sequential probability ratio test to a sample-size range."""
    p0, p1 = _open_probability(p0, "p0"), _open_probability(p1, "p1")
    alpha, beta = _open_probability(alpha, "alpha"), _open_probability(beta, "beta")
    if p0 >= p1:
        raise ValueError("p0 must be less than p1")
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be less than 1")
    endpoints = _integer_vector([min_sample_size, max_sample_size], "sample sizes")
    if not 1 <= endpoints[0] <= endpoints[1]:
        raise ValueError("require 1 <= min_sample_size <= max_sample_size")
    sizes = np.arange(endpoints[0], endpoints[1] + 1)
    difference = p1 - p0
    failure_ratio = difference / (1 - p0)
    log_failure = np.log1p(-failure_ratio) if failure_ratio <= 0.5 else np.log1p(-p1) - np.log1p(-p0)
    log_success = np.log1p(difference / p0) if difference <= p0 else np.log(p1) - np.log(p0)
    response_gain = log_success - log_failure
    slope = -log_failure / response_gain
    upper_intercept = (np.log1p(-beta) - np.log(alpha)) / response_gain
    lower_intercept = (np.log(beta) - np.log1p(-alpha)) / response_gain
    lower = np.maximum(-1, np.floor(slope * sizes + lower_intercept)).astype(np.intp)
    upper = np.ceil(slope * sizes + upper_intercept).astype(np.intp)
    testable = (lower >= 0) | (upper <= sizes)
    # Retain the truncation even when no boundary can be crossed at any look.
    testable[-1] = True
    lower, upper = lower[testable], upper[testable]
    characteristics = binomial_crossing_probabilities(sizes[testable], lower, upper, [p0, p1])
    return BinomialSPRT(lower, upper, float(lower_intercept), float(upper_intercept), float(slope), characteristics)
