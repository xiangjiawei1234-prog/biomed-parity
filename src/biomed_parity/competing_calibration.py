"""Fixed-horizon calibration of survival and competing-risk predictions."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _numeric_vector


@dataclass(frozen=True)
class RiskCalibration:
    horizon: float
    predicted: NDArray[np.float64]
    observed: NDArray[np.float64]
    observations: NDArray[np.int64]
    cutpoints: NDArray[np.float64] | None
    pseudo_values: NDArray[np.float64] | None


def _outcomes(time, status, horizon, cause):
    times = _numeric_vector(time, "time")
    events = _numeric_vector(status, "status")
    if len(times) != len(events) or np.any(times < 0):
        raise ValueError("time and status must be aligned with nonnegative follow-up")
    if np.any((events < 0) | (events != np.floor(events))):
        raise ValueError("status must contain nonnegative integer codes")
    if not isinstance(cause, Integral) or isinstance(cause, bool) or cause < 1:
        raise ValueError("cause must be a positive integer")
    if not isinstance(horizon, Real) or not 0 <= horizon <= times.max():
        raise ValueError("horizon must be within [0, maximum follow-up]")
    return times, events


def _risk_counts(times, events, cause):
    knots, index, group_size = np.unique(times, return_inverse=True, return_counts=True)
    sample_size = len(times)
    risk_set = sample_size - np.r_[0, np.cumsum(group_size[:-1])]
    failures = np.bincount(index[events > 0], minlength=len(knots))
    targets = np.bincount(index[events == cause], minlength=len(knots))
    return knots, index, risk_set, failures, targets


def _cumulative_incidence(failure_hazard, target_hazard):
    before = np.r_[1.0, np.cumprod(1 - failure_hazard)]
    return np.r_[0.0, np.cumsum(before[:-1] * target_hazard)], before


def _local_risk(times, events, cause, horizon):
    # prodlim reports unavailable extrapolation beyond the local follow-up.
    if not len(times) or max(times) < horizon:
        return np.nan
    knots, _, risk, failures, targets = _risk_counts(times, events, cause)
    end = np.searchsorted(knots, horizon, side="right")
    cumulative, _ = _cumulative_incidence(failures[:end] / risk[:end], targets[:end] / risk[:end])
    return cumulative[-1]


def _pseudovalues(times, events, cause, horizon):
    if len(times) < 2:
        raise ValueError("jackknife calibration requires at least two observations")
    knots, index, risk, failures, targets = _risk_counts(times, events, cause)
    end = np.searchsorted(knots, horizon, side="right")
    risk, failures, targets = risk[:end], failures[:end], targets[:end]
    failure_hazard, target_hazard = failures / risk, targets / risk
    cumulative, _ = _cumulative_incidence(failure_hazard, target_hazard)

    # Before a deleted individual's stop time, each risk set loses one member.
    # After that stop the full-cohort hazards apply. A backward recursion saves
    # the conditional contribution of those later time points.
    future = np.zeros(end + 1)
    for j in range(end - 1, -1, -1):
        future[j] = target_hazard[j] + (1 - failure_hazard[j]) * future[j + 1]
    other_risk = risk - 1
    reduced_failure = np.divide(
        failures, other_risk, where=other_risk > 0, out=np.zeros(end),
    )
    reduced_target = np.divide(
        targets, other_risk, where=other_risk > 0, out=np.zeros(end),
    )
    reduced_incidence, reduced_survival = _cumulative_incidence(reduced_failure, reduced_target)

    omit_one = np.full(len(times), reduced_incidence[-1])
    inside = index < end
    at = index[inside]
    support = other_risk[at]
    removed_failures = failures[at] - (events[inside] > 0)
    removed_targets = targets[at] - (events[inside] == cause)
    lost = np.divide(removed_failures, support, where=support > 0, out=np.zeros(len(at)))
    cause_risk = np.divide(removed_targets, support, where=support > 0, out=np.zeros(len(at)))
    omit_one[inside] = reduced_incidence[at] + reduced_survival[at] * (
        cause_risk + (1 - lost) * future[at + 1]
    )
    # If deleting the only at-risk failure, prodlim's standalone jackknife
    # reports an undefined pseudo-value at its failure time.
    deleted_failure = np.flatnonzero(inside)[(support == 0) & (events[inside] > 0)]
    omit_one[deleted_failure] = np.nan
    n = len(times)
    return n * cumulative[-1] - (n - 1) * omit_one


def jackknife_cumulative_incidence(
    time: ArrayLike,
    status: ArrayLike,
    horizon: float,
    *,
    cause: int = 1,
) -> NDArray[np.float64]:
    """Return row-aligned Aalen–Johansen leave-one-out pseudo-values."""
    times, events = _outcomes(time, status, horizon, cause)
    return _pseudovalues(times, events, cause, horizon)


def _quantile_groups(risks, groups):
    if isinstance(groups, Integral) and not isinstance(groups, bool):
        if groups < 1:
            raise ValueError("groups must be a positive quantile count")
        breaks = np.unique(np.quantile(risks, np.linspace(0, 1, groups + 1)))
    else:
        breaks = _numeric_vector(groups, "groups")
        if (
            len(breaks) < 2 or np.any(np.diff(breaks) <= 0)
            or not 0 <= breaks[0] <= risks.min() <= risks.max() <= breaks[-1] <= 1
        ):
            raise ValueError("group cutpoints must increase and span the predictions in [0, 1]")

    unique_risk, labels = np.unique(risks, return_inverse=True)
    if len(unique_risk) <= len(breaks):
        groups = tuple(np.flatnonzero(labels == i) for i in range(len(unique_risk)))
    else:
        bins = np.searchsorted(breaks[1:], risks, side="left")
        groups = tuple(np.flatnonzero(bins == k) for k in range(len(breaks) - 1))
    predicted = np.array([risks[rows].mean() if len(rows) else np.nan for rows in groups])
    return predicted, groups, breaks


def _neighborhoods(risks, bandwidth, digits):
    if bandwidth is None or not isinstance(bandwidth, Real) or not 0 <= bandwidth < np.inf:
        raise ValueError("method='nne' requires a finite nonnegative bandwidth")
    if digits is not None and (
        not isinstance(digits, Integral) or isinstance(digits, bool) or not 0 <= digits <= 15
    ):
        raise ValueError("round_digits must be None or an integer between 0 and 15")
    if bandwidth >= 1:
        return np.array([risks.mean()]), (np.arange(len(risks)),)

    rounded = np.round(risks, digits) if digits else risks
    values, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(rounded, kind="stable")
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    radius = int(bandwidth * len(risks))
    first = starts[np.searchsorted(cumulative, cumulative - radius, side="left")]
    last = cumulative[np.searchsorted(cumulative, cumulative + radius, side="right") - 1]
    return values, tuple(order[a:b] for a, b in zip(first, last, strict=True))


def calibrate_competing_risks(
    time: ArrayLike,
    status: ArrayLike,
    risks: ArrayLike,
    horizon: float,
    *,
    cause: int = 1,
    method: Literal["quantile", "nne"] = "quantile",
    estimator: Literal["local", "jackknife"] = "local",
    groups: int | ArrayLike = 10,
    bandwidth: float | None = None,
    round_digits: int | None = 2,
) -> RiskCalibration:
    """Compare predictions with locally observed cumulative incidence.

    With method='quantile', use prediction quantiles or explicit cutpoints.
    With method='nne', supply the neighborhood radius as a sample fraction.
    'local' fits Aalen–Johansen within each group; 'jackknife' averages
    full-cohort pseudo-values within the group. Values beyond the observed
    follow-up of a local group are undefined.
    """
    times, events = _outcomes(time, status, horizon, cause)
    risk = _numeric_vector(risks, "risks")
    if len(risk) != len(times) or np.any((risk < 0) | (risk > 1)):
        raise ValueError("risks must contain one probability per observation")
    if method == "quantile":
        predicted, neighborhoods, breaks = _quantile_groups(risk, groups)
    elif method == "nne":
        predicted, neighborhoods = _neighborhoods(risk, bandwidth, round_digits)
        breaks = None
    else:
        raise ValueError("method must be 'quantile' or 'nne'")

    if estimator == "local":
        pseudo = None
        observed = [_local_risk(times[rows], events[rows], cause, horizon) for rows in neighborhoods]
    elif estimator == "jackknife":
        pseudo = _pseudovalues(times, events, cause, horizon)
        observed = [pseudo[rows].mean() if len(rows) else np.nan for rows in neighborhoods]
    else:
        raise ValueError("estimator must be 'local' or 'jackknife'")
    return RiskCalibration(
        float(horizon), predicted, np.asarray(observed), np.array([len(p) for p in neighborhoods]),
        breaks, pseudo,
    )
