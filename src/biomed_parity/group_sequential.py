"""Boundaries and crossing probabilities under the canonical group-sequential model."""

from dataclasses import dataclass
from functools import lru_cache
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtri, ndtri_exp, roots_legendre

_SQRT_2PI = np.sqrt(2 * np.pi)
_LOG_2 = np.log(2)
_MIN_Z = -12.0


@dataclass(frozen=True)
class EfficacyBoundaries:
    """Each vector corresponds to the planned information fractions."""

    information_fractions: NDArray[np.float64]
    z_boundaries: NDArray[np.float64]
    nominal_pvalues: NDArray[np.float64]
    cumulative_alpha: NDArray[np.float64]
    incremental_alpha: NDArray[np.float64]
    null_crossing_probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class CrossingProbabilities:
    """First crossing and continuation probabilities at each reported look."""

    information_fractions: NDArray[np.float64]
    upper: NDArray[np.float64]
    lower: NDArray[np.float64]
    continuing: NDArray[np.float64]

    @property
    def upper_total(self) -> float:
        return float(self.upper.sum())


@dataclass(frozen=True)
class TwoBoundaryDesign:
    """Symmetric efficacy or asymmetric efficacy and futility design."""

    information_fractions: NDArray[np.float64]
    lower_boundaries: NDArray[np.float64]
    upper_boundaries: NDArray[np.float64]
    cumulative_alpha: NDArray[np.float64]
    cumulative_beta: NDArray[np.float64] | None = None
    alternative_effect: float | None = None
    binding_futility: bool | None = None


def _spending(information, alpha, spending, parameter):
    if spending == "hsd":
        gamma = -4.0 if parameter is None else parameter
        if not -40 <= gamma <= 40:
            raise ValueError("HSD gamma must be in [-40, 40]")
        if gamma == 0:
            return alpha * information
        return alpha * (np.expm1(-gamma * information) / np.expm1(-gamma))
    if spending == "ldof":
        # gsDesign 3.11.0 substitutes rho=1 outside its supported range.
        rho = 1.0 if parameter is None or not 0.005 <= parameter <= 20 else parameter
        threshold = -ndtri_exp(np.log(alpha) - _LOG_2)
        with np.errstate(divide="ignore", over="ignore"):
            return np.exp(_LOG_2 + log_ndtr(-threshold / information ** (rho / 2)))
    raise ValueError("spending must be 'hsd' or 'ldof'")


def _normal_pdf(z):
    return np.exp(-0.5 * z * z) / _SQRT_2PI


@lru_cache(maxsize=24)
def _quadrature_rule(order):
    return roots_legendre(order)


def _grid(boundary, incoming_sd, outgoing_sd, lowest=_MIN_Z):
    span = boundary - lowest
    # Resolve the narrower of the incoming and outgoing Gaussian kernels.
    order = max(96, int(np.ceil(2 * span / min(incoming_sd, outgoing_sd))))
    if order > 2048:
        raise ArithmeticError("information looks are too close for accurate quadrature")
    nodes, weights = _quadrature_rule(order)
    return lowest + 0.5 * span * (nodes + 1), 0.5 * span * weights


def _increment_sd(previous, current):
    return np.sqrt((current - previous) / current)


def _information_vector(information_fractions):
    information = np.asarray(information_fractions)
    if information.ndim != 1 or not information.size or information.dtype.kind not in "iuf":
        raise ValueError("information_fractions must be a nonempty numeric vector")
    information = information.astype(float)
    if (
        not np.isfinite(information).all()
        or information[0] <= 0
        or information[-1] != 1
        or np.any(np.diff(information) <= 0)
    ):
        raise ValueError("information_fractions must strictly increase from above 0 to 1")
    return information


def _design_spending(information, budget, name, spending, parameter):
    if not isinstance(budget, Real) or not 0 < budget < 0.5:
        raise ValueError(f"{name} must be in (0, 0.5)")
    if parameter is not None and (not isinstance(parameter, Real) or not np.isfinite(parameter)):
        raise ValueError("parameter must be a finite real number")
    cumulative = _spending(information, float(budget), spending, parameter)
    cumulative[-1] = budget
    return cumulative


def efficacy_boundaries(
    information_fractions: ArrayLike,
    *,
    alpha: float = 0.025,
    spending: str = "hsd",
    parameter: float | None = None,
) -> EfficacyBoundaries:
    """Calibrate upper Z boundaries to a one-sided cumulative alpha budget.

    Fractions are strictly increasing in (0, 1] and terminate at 1. The
    spending choices are Hwang–Shih–DeCani ('hsd', gamma defaults to -4)
    and Lan–DeMets O'Brien–Fleming ('ldof', rho defaults to 1).
    """
    information = _information_vector(information_fractions)
    cumulative = _design_spending(information, alpha, "alpha", spending, parameter)
    increments = np.diff(cumulative, prepend=0)
    active = np.flatnonzero(increments > 0)
    boundaries = np.full(information.size, np.inf)
    crossing = np.zeros(information.size)
    previous_time = 0.0
    for position, look in enumerate(active):
        t = information[look]
        target = increments[look]
        incoming_sd = _increment_sd(previous_time, t)
        correlation = np.sqrt(previous_time / t)
        independent_bound = -ndtri(target)
        if position == 0:
            boundary = independent_bound
            crossing[look] = np.exp(log_ndtr(-boundary))
        else:
            conditional_means = correlation * previous_nodes

            def probability(z):
                tails = np.exp(log_ndtr((conditional_means - z) / incoming_sd))
                return float(np.sum(previous_mass * tails))

            boundary = brentq(
                lambda z: probability(z) - target,
                _MIN_Z,
                independent_bound + 1.0,
                xtol=1e-12,
            )
            crossing[look] = probability(boundary)
        boundaries[look] = boundary

        if position + 1 == len(active):
            break
        next_time = information[active[position + 1]]
        nodes, weights = _grid(boundary, incoming_sd, _increment_sd(t, next_time))
        if position == 0:
            density = _normal_pdf(nodes)
        else:
            z = (nodes[:, None] - conditional_means) / incoming_sd
            density = np.einsum("ij,j->i", _normal_pdf(z), previous_mass) / incoming_sd
        previous_mass = weights * density
        if abs(previous_mass.sum() - (1 - cumulative[look])) > 5e-10:
            raise ArithmeticError("quadrature failed to recover the continuing probability")
        previous_nodes = nodes
        previous_time = t

    return EfficacyBoundaries(
        information_fractions=information,
        z_boundaries=boundaries,
        nominal_pvalues=np.exp(log_ndtr(-boundaries)),
        cumulative_alpha=cumulative,
        incremental_alpha=increments,
        null_crossing_probabilities=crossing,
    )


@dataclass(frozen=True)
class _Stage:
    means: NDArray[np.float64]
    mass: NDArray[np.float64]
    sd: float

    def tail(self, bound, upper):
        offset = self.means - bound if upper else bound - self.means
        return float(np.sum(self.mass * np.exp(log_ndtr(offset / self.sd))))

    def quantile(self, probability, upper):
        if probability == 0:
            return np.inf if upper else -np.inf
        if probability >= self.mass.sum():
            return -np.inf if upper else np.inf
        quantile = -ndtri(probability) if upper else ndtri(probability)
        if len(self.mass) == 1:
            return self.means[0] + self.sd * quantile
        limit = np.abs(self.means).max() + max(12.0, abs(quantile) + 1)
        return brentq(lambda z: self.tail(z, upper) - probability, -limit, limit, xtol=1e-12)

    def restrict(self, lower, upper, outgoing_sd, continuing):
        low = max(lower, _MIN_Z)
        high = min(upper, -_MIN_Z)
        if low >= high or not self.mass.size:
            nodes = mass = np.empty(0)
        else:
            nodes, weights = _grid(high, self.sd, outgoing_sd, low)
            z = (nodes[:, None] - self.means) / self.sd
            density = np.einsum("ij,j->i", _normal_pdf(z), self.mass) / self.sd
            mass = weights * density
        if abs(mass.sum() - continuing) > 5e-10:
            raise ArithmeticError("quadrature failed to recover the continuing probability")
        return nodes, mass


def _next_stage(t, previous, nodes, mass):
    return _Stage(np.sqrt(previous / t) * nodes, mass, _increment_sd(previous, t))


def _bound_vector(values, n, name):
    bounds = np.asarray(values)
    if bounds.ndim != 1 or len(bounds) != n or bounds.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector with one value per look")
    bounds = bounds.astype(float)
    if np.isnan(bounds).any():
        raise ValueError(f"{name} must have no NaN values")
    return bounds


def _crossings(information, lower, upper):
    previous = 0.0
    nodes = np.zeros(1)
    mass = np.ones(1)
    up, down, continuing = np.zeros((3, len(information)))
    continuing.fill(1)
    active = np.flatnonzero(~np.isneginf(lower) | ~np.isposinf(upper))
    for position, i in enumerate(active):
        t = information[i]
        stage = _next_stage(t, previous, nodes, mass)
        up[i] = stage.tail(upper[i], upper=True)
        down[i] = stage.tail(lower[i], upper=False)
        remaining = max(0.0, mass.sum() - up[i] - down[i])
        next_look = active[position + 1] if position + 1 < len(active) else len(information)
        continuing[i:next_look] = remaining
        if next_look < len(information):
            outgoing_sd = _increment_sd(t, information[next_look])
            nodes, mass = stage.restrict(lower[i], upper[i], outgoing_sd, remaining)
            previous = t
    return up, down, continuing


def _crossing_inputs(information_fractions, lower_boundaries, upper_boundaries, effect, respect_lower):
    information = _information_vector(information_fractions)
    lower = _bound_vector(lower_boundaries, len(information), "lower_boundaries")
    upper = _bound_vector(upper_boundaries, len(information), "upper_boundaries")
    if np.any(lower > upper):
        raise ValueError("lower boundaries must not exceed upper boundaries")
    if not isinstance(effect, Real) or not np.isfinite(effect):
        raise ValueError("effect must be a finite real number")
    if not isinstance(respect_lower, (bool, np.bool_)):
        raise ValueError("respect_lower must be boolean")
    if not respect_lower:
        lower.fill(-np.inf)
    return information, lower, upper


def crossing_probabilities(
    information_fractions: ArrayLike,
    lower_boundaries: ArrayLike,
    upper_boundaries: ArrayLike,
    *,
    effect: float = 0.0,
    respect_lower: bool = True,
) -> CrossingProbabilities:
    """Evaluate first crossings for a specified final standardized mean Z.

    With respect_lower=False all lower stopping bounds are ignored.
    Upper and lower bounds may use infinities to disable stopping.
    """
    info, lower, upper = _crossing_inputs(
        information_fractions, lower_boundaries, upper_boundaries, effect, respect_lower
    )
    means = effect * np.sqrt(info)
    return CrossingProbabilities(info, *_crossings(info, lower - means, upper - means))


def conditional_power(
    information_fractions: ArrayLike,
    lower_boundaries: ArrayLike,
    upper_boundaries: ArrayLike,
    *,
    look: int,
    z: float,
    effect: float = 0.0,
    respect_lower: bool = True,
) -> CrossingProbabilities:
    """Future first crossings given a zero-based interim look and observed Z.

    The trial is assumed to continue through that look. Effect is the
    hypothesized marginal mean of the final Z. Use upper_total for the
    probability of any future upper crossing.
    """
    info, lower, upper = _crossing_inputs(
        information_fractions, lower_boundaries, upper_boundaries, effect, respect_lower
    )
    if (
        not isinstance(look, (int, np.integer))
        or isinstance(look, (bool, np.bool_))
        or not 0 <= look < len(info) - 1
    ):
        raise ValueError("look must be the zero-based index of a nonfinal look")
    if not isinstance(z, Real) or not np.isfinite(z):
        raise ValueError("z must be a finite real number")
    future = info[look + 1 :]
    increment = future - info[look]
    mean = (np.sqrt(info[look]) * z + effect * increment) / np.sqrt(future)
    sd = np.sqrt(increment / future)
    result = _crossings(
        increment / increment[-1], (lower[look + 1 :] - mean) / sd,
        (upper[look + 1 :] - mean) / sd,
    )
    return CrossingProbabilities(future, *result)


def symmetric_boundaries(
    information_fractions: ArrayLike,
    *,
    alpha: float = 0.025,
    spending: str = "hsd",
    parameter: float | None = None,
) -> TwoBoundaryDesign:
    """Calibrate mirrored efficacy boundaries for type I error 2 * alpha."""
    info = _information_vector(information_fractions)
    cumulative = _design_spending(info, alpha, "alpha", spending, parameter)
    increments = np.diff(cumulative, prepend=0)
    active = np.flatnonzero(increments > 0)
    bounds = np.full(len(info), np.inf)
    previous = 0.0
    nodes = np.zeros(1)
    mass = np.ones(1)
    for position, i in enumerate(active):
        t = info[i]
        stage = _next_stage(t, previous, nodes, mass)
        bounds[i] = stage.quantile(increments[i], upper=True)
        if position + 1 < len(active):
            outgoing_sd = _increment_sd(t, info[active[position + 1]])
            nodes, mass = stage.restrict(-bounds[i], bounds[i], outgoing_sd, 1 - 2 * cumulative[i])
            previous = t
    return TwoBoundaryDesign(info, -bounds, bounds, cumulative)


def _futility_candidate(info, alpha_increment, beta_increment, effect, fixed_upper):
    upper = np.full(len(info), np.inf)
    lower = np.full(len(info), -np.inf)
    previous = 0.0
    zero = np.zeros(1)
    one = np.ones(1)
    null_nodes = alternative_nodes = zero
    null_mass = alternative_mass = one
    alternative_means = effect * np.sqrt(info)
    active = np.flatnonzero((alpha_increment > 0) | (beta_increment > 0))
    for position, i in enumerate(active):
        t = info[i]
        alt = _next_stage(t, previous, alternative_nodes, alternative_mass)
        lower[i] = alt.quantile(beta_increment[i], upper=False) + alternative_means[i]
        if fixed_upper is None:
            null = _next_stage(t, previous, null_nodes, null_mass)
            upper[i] = null.quantile(alpha_increment[i], upper=True)
        else:
            upper[i] = fixed_upper[i]

        if i + 1 == len(info):
            return lower[i] - upper[i], lower, upper
        if lower[i] >= upper[i]:
            # This effect spends the alternative continuation mass too early.
            return np.inf, None, None
        outgoing_sd = _increment_sd(t, info[active[position + 1]])
        alt_low = lower[i] - alternative_means[i]
        alt_up = upper[i] - alternative_means[i]
        alt_remaining = alternative_mass.sum() - alt.tail(alt_low, upper=False) - alt.tail(alt_up, upper=True)
        alternative_nodes, alternative_mass = alt.restrict(alt_low, alt_up, outgoing_sd, alt_remaining)
        if fixed_upper is None:
            null_remaining = null_mass.sum() - null.tail(lower[i], upper=False) - null.tail(upper[i], upper=True)
            null_nodes, null_mass = null.restrict(lower[i], upper[i], outgoing_sd, null_remaining)
        previous = t
    raise AssertionError("at least one information fraction is required")


def efficacy_futility_boundaries(
    information_fractions: ArrayLike,
    *,
    alpha: float = 0.025,
    beta: float = 0.1,
    spending: str = "hsd",
    parameter: float | None = None,
    futility_spending: str = "hsd",
    futility_parameter: float | None = None,
    binding: bool = False,
) -> TwoBoundaryDesign:
    """Calibrate alpha and beta spending with one-sided efficacy and futility.

    The estimand is the final standardized alternative mean yielding power
    1 - beta. For binding=False, efficacy level is calibrated ignoring
    lower stops under the null. Futility spending is imposed under the
    alternative in both cases. HSD futility gamma defaults to -2.
    """
    info = _information_vector(information_fractions)
    if not isinstance(binding, (bool, np.bool_)):
        raise ValueError("binding must be boolean")
    cumulative_alpha = _design_spending(info, alpha, "alpha", spending, parameter)
    if futility_spending == "hsd" and futility_parameter is None:
        futility_parameter = -2.0
    cumulative_beta = _design_spending(info, beta, "beta", futility_spending, futility_parameter)
    a = np.diff(cumulative_alpha, prepend=0)
    b = np.diff(cumulative_beta, prepend=0)
    if a[-1] <= 0 or b[-1] <= 0:
        raise ValueError("both spending functions must have positive final increments")
    fixed_upper = None if binding else efficacy_boundaries(
        info, alpha=alpha, spending=spending, parameter=parameter
    ).z_boundaries

    def candidate(effect):
        return _futility_candidate(info, a, b, effect, fixed_upper)

    high = max(4.0, -ndtri(alpha) - ndtri(beta) + 2)
    for _ in range(6):
        if candidate(high)[0] > 0:
            break
        high *= 2
    else:
        raise ArithmeticError("unable to bracket the alternative effect")
    effect = brentq(lambda effect: candidate(effect)[0], 0, high, xtol=1e-11)
    _, lower, upper = candidate(effect)
    if lower is None or not np.isclose(lower[-1], upper[-1], rtol=0, atol=1e-8):
        raise ArithmeticError("unable to calibrate coincident final boundaries")
    lower[-1] = upper[-1]
    return TwoBoundaryDesign(info, lower, upper, cumulative_alpha, cumulative_beta, effect, binding)
