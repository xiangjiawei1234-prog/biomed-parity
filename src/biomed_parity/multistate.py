"""Aalen–Johansen transition probabilities from cumulative hazard trajectories."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _numeric_vector


@dataclass(frozen=True)
class TransitionProbabilities:
    """The first axis of each output indexes the returned times.

    The remaining probability and standard error axes are origin and
    destination. Covariance uses column-major indexing of these pairs.
    """

    times: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    standard_errors: NDArray[np.float64] | None
    covariance: NDArray[np.float64] | None


def _covariance_increments(cumulative, shape, keep):
    cumulative = np.asarray(cumulative, dtype=float)
    if cumulative.shape != shape or not np.isfinite(cumulative).all():
        raise ValueError("cumulative_covariance must have shape (times, transitions, transitions)")
    if not np.allclose(cumulative, cumulative.swapaxes(1, 2), rtol=1e-12, atol=1e-14):
        raise ValueError("cumulative_covariance matrices must be symmetric")
    cumulative = cumulative[keep]
    return np.diff(cumulative, axis=0, prepend=np.zeros((1, *shape[1:])))


def _jump_matrix(hazards, origins, destinations, identity, time):
    jump = identity.copy()
    jump[origins, destinations] = hazards
    totals = np.bincount(origins, weights=hazards, minlength=len(identity))
    if np.any(totals > 1 + 1e-12):
        raise ValueError(f"a state's outgoing hazard jumps exceed 1 at time {time:g}")
    jump[np.diag_indices_from(jump)] = np.maximum(0, 1 - totals)
    return jump


def _congruence(mapping, cov):
    transformed = np.einsum("ij,jk->ik", mapping, cov)
    return np.einsum("ij,kj->ik", transformed, mapping)


def _step_covariance(variance, increment, change, identity, jump, previous, current, forward, method):
    if forward:
        propagation = np.kron(jump.T, identity)
        noise = np.kron(identity, current if method == "aalen" else previous)
    else:
        propagation = np.kron(identity, jump)
        noise = np.kron(previous.T, jump if method == "aalen" else identity)
    derivative = noise @ change
    return _congruence(propagation, variance) + _congruence(derivative, increment)


def probtrans(
    times: ArrayLike,
    transitions: ArrayLike,
    cumulative_hazards: ArrayLike,
    *,
    predt: float = 0,
    direction: str = "forward",
    n_states: int | None = None,
    cumulative_covariance: ArrayLike | None = None,
    method: str = "aalen",
    covariance: bool = False,
) -> TransitionProbabilities:
    """Compute all origin-to-destination probabilities on a shared time grid.

    Transitions are distinct zero-based (origin, destination) state pairs.
    Cumulative hazards are time-by-transition and nondecreasing within
    each column. Rows at equal times are consolidated before any jump.
    For a forward result, predt fixes the starting time. For a fixed
    horizon result it fixes the endpoint, with starting times on the grid.
    Standard errors require cumulative covariance of the supplied hazards.
    """
    times = _numeric_vector(times, "times")
    if np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError("times must be nonnegative and sorted")
    if not isinstance(predt, Real) or not np.isfinite(predt) or predt < 0:
        raise ValueError("predt must be finite and nonnegative")
    if direction not in {"forward", "fixedhorizon"}:
        raise ValueError("direction must be 'forward' or 'fixedhorizon'")
    if method not in {"aalen", "greenwood"}:
        raise ValueError("method must be 'aalen' or 'greenwood'")

    transitions = np.asarray(transitions)
    if (
        transitions.ndim != 2
        or transitions.shape[1] != 2
        or not len(transitions)
        or transitions.dtype.kind not in "iu"
        or np.any(transitions < 0)
    ):
        raise ValueError("transitions must contain zero-based nonnegative integer state pairs")
    origins, destinations = transitions.T.astype(np.intp)
    if np.any(origins == destinations) or len(np.unique(transitions, axis=0)) != len(transitions):
        raise ValueError("transitions must be distinct and may not contain self-transitions")
    minimum_states = int(transitions.max()) + 1
    if n_states is None:
        n_states = minimum_states
    elif not isinstance(n_states, Integral) or n_states < minimum_states:
        raise ValueError("n_states must include every state in transitions")

    hazards = np.asarray(cumulative_hazards)
    if hazards.dtype.kind not in "iuf":
        raise ValueError("cumulative_hazards must be numeric")
    hazards = hazards.astype(float)
    m = len(transitions)
    if hazards.shape != (len(times), m) or not np.isfinite(hazards).all():
        raise ValueError("cumulative_hazards must be finite with shape (times, transitions)")
    if np.any(hazards < 0) or np.any(np.diff(hazards, axis=0) < 0):
        raise ValueError("each cumulative hazard must be nonnegative and nondecreasing")

    if covariance and cumulative_covariance is None:
        raise ValueError("cumulative_covariance is required for a covariance result")
    keep = np.r_[times[:-1] != times[1:], True]
    uncertainty = None
    if cumulative_covariance is not None:
        uncertainty = _covariance_increments(cumulative_covariance, (len(times), m, m), keep)
    times = times[keep]
    increments = np.diff(hazards[keep], axis=0, prepend=np.zeros((1, m)))
    identity = np.eye(n_states)
    if uncertainty is not None:
        # Unit changes in each off-diagonal edge affect two generator entries.
        change = np.zeros((n_states**2, m))
        change[origins + n_states * destinations, np.arange(m)] = 1
        change[origins + n_states * origins, np.arange(m)] = -1
        variance = np.zeros((n_states**2, n_states**2))

    forward = direction == "forward"
    if forward:
        selected = np.flatnonzero(times > predt)
        result_times = np.r_[predt, times[selected]]
        steps = zip(selected, range(1, len(result_times)))
    else:
        selected = np.flatnonzero((times > 0) & (times <= predt))
        result_times = np.unique(np.r_[0, times[selected], predt])
        steps = zip(selected[::-1], range(len(selected) - 1, -1, -1))

    probabilities = np.broadcast_to(identity, (len(result_times), n_states, n_states)).copy()
    errors = np.zeros_like(probabilities) if uncertainty is not None else None
    covariances = np.zeros((len(result_times), n_states**2, n_states**2)) if covariance else None
    previous = identity
    for i, position in steps:
        jump = _jump_matrix(increments[i], origins, destinations, identity, times[i])
        current = previous @ jump if forward else jump @ previous
        probabilities[position] = current
        if uncertainty is not None:
            variance = _step_covariance(
                variance, uncertainty[i], change, identity, jump, previous, current, forward, method,
            )
            diagonal = variance.diagonal().reshape(identity.shape, order="F")
            errors[position] = np.sqrt(np.where(diagonal < -1e-12, np.nan, np.maximum(diagonal, 0)))
            if covariance:
                covariances[position] = variance
        previous = current
    return TransitionProbabilities(result_times, probabilities, errors, covariances)
