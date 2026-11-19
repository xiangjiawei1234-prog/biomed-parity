"""One-sided efficacy boundaries under the canonical group-sequential model."""

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


def _grid(boundary, incoming_sd, outgoing_sd):
    span = boundary - _MIN_Z
    # Resolve the narrower of the incoming and outgoing Gaussian kernels.
    order = max(96, int(np.ceil(2 * span / min(incoming_sd, outgoing_sd))))
    if order > 2048:
        raise ArithmeticError("information looks are too close for accurate quadrature")
    nodes, weights = _quadrature_rule(order)
    return _MIN_Z + 0.5 * span * (nodes + 1), 0.5 * span * weights


def _increment_sd(previous, current):
    return np.sqrt((current - previous) / current)


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
    if not isinstance(alpha, Real) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if parameter is not None and (not isinstance(parameter, Real) or not np.isfinite(parameter)):
        raise ValueError("parameter must be a finite real number")

    cumulative = _spending(information, float(alpha), spending, parameter)
    cumulative[-1] = alpha
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
