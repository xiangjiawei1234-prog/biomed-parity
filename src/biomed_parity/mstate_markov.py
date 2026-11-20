"""Log-rank tests of the Markov assumption for multi-state transition data."""

from dataclasses import dataclass
from numbers import Integral
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _encode_labels, _numeric_vector
from .mstate_fit import _counting_data
from .mstate_prep import MultiStateRecords


@dataclass(frozen=True)
class MarkovTestResult:
    """Observed and wild-bootstrap traces for one transition."""

    transition: int
    from_state: int
    to_state: int
    qualifying_states: NDArray[np.intp]
    grid: NDArray[np.float64]
    n_subjects: int
    n_events_after: NDArray[np.intp]
    event_subject_ids: NDArray
    event_times: NDArray[np.float64]
    score_trace: NDArray[np.float64]
    z_trace: NDArray[np.float64]
    covariance: NDArray[np.float64]
    chi_squared_trace: NDArray[np.float64]
    state_statistics: NDArray[np.float64]
    overall_statistic: float
    bootstrap_z_trace: NDArray[np.float64]
    bootstrap_chi_squared_trace: NDArray[np.float64]
    bootstrap_state_statistics: NDArray[np.float64]
    bootstrap_overall_statistics: NDArray[np.float64]
    state_pvalues: NDArray[np.float64] | None
    overall_pvalue: float | None
    z_quantiles: NDArray[np.float64] | None


@dataclass(frozen=True)
class _Episodes:
    labels: NDArray
    subject: NDArray[np.intp]
    origin: NDArray[np.intp]
    start: NDArray[np.float64]
    stop: NDArray[np.float64]
    outcome: NDArray[np.intp]


def _episodes(records):
    if not isinstance(records, MultiStateRecords):
        raise TypeError("records must be a MultiStateRecords object")
    drop = records.t_stop <= records.t_start
    if drop.any():
        warn(f"removed {drop.sum()} rows with nonpositive interval duration", stacklevel=3)
    keep = ~drop
    start, stop, event, edge_index, edges, _ = _counting_data(
        records.t_start[keep], records.t_stop[keep], records.status[keep],
        records.transition[keep], records.transitions, records.n_states, timefix=False,
    )
    labels, subjects = _encode_labels(records.id[keep], None, len(start), "id")
    if not len(subjects):
        raise ValueError("records contain no positive-duration intervals")
    origin = edges[edge_index, 0]
    order = np.lexsort((stop, origin, start, subjects))
    subjects, origin, start, stop, event, edge_index = [
        values[order] for values in (subjects, origin, start, stop, event, edge_index)
    ]
    # Multiple outgoing transition-risk rows describe one observed state episode.
    cuts = np.flatnonzero(np.r_[
        True, (np.diff(subjects) != 0) | (np.diff(origin) != 0)
        | (start[1:] != start[:-1]) | (stop[1:] != stop[:-1]),
    ])
    if np.any(np.add.reduceat(event.astype(int), cuts) > 1):
        raise ValueError("each state episode may have at most one observed transition")
    outcome = np.maximum.reduceat(np.where(event, edge_index, -1), cuts)
    subjects, origin, start, stop = (values[cuts] for values in (subjects, origin, start, stop))
    if np.any((subjects[1:] == subjects[:-1]) & (start[1:] < stop[:-1])):
        raise ValueError("state episodes within a subject must not overlap")
    return _Episodes(np.asarray(labels), subjects, origin, start, stop, outcome)


def _qualifying_states(edges, n_states, origin):
    reachable = np.zeros(n_states, dtype=bool)
    reachable[origin] = True
    origins, destinations = edges.T
    while True:
        predecessors = origins[reachable[destinations]]
        if reachable[predecessors].all():
            return np.flatnonzero(reachable)
        reachable[predecessors] = True


def _bootstrap_multipliers(multipliers, n_bootstrap, distribution, random_state, events):
    if n_bootstrap is not None and (
        not isinstance(n_bootstrap, Integral) or isinstance(n_bootstrap, (bool, np.bool_)) or n_bootstrap < 0
    ):
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if distribution not in ("poisson", "normal"):
        raise ValueError("distribution must be 'poisson' or 'normal'")
    if multipliers is not None:
        samples = np.asarray(multipliers)
        if (
            samples.ndim != 2 or samples.shape[1] != events or samples.dtype.kind not in "iuf"
            or not np.isfinite(samples).all()
        ):
            raise ValueError("multipliers must be a finite matrix with one column per observed transition event")
        if n_bootstrap is not None and n_bootstrap != len(samples):
            raise ValueError("n_bootstrap must match the number of multiplier rows")
        if random_state is not None:
            raise ValueError("use either multipliers or random_state")
        return samples.astype(float)
    shape = (1000 if n_bootstrap is None else n_bootstrap, events)
    rng = np.random.default_rng(random_state)
    if distribution == "poisson":
        return rng.poisson(1, size=shape).astype(float) - 1
    return rng.standard_normal(size=shape)


def _joint_trace(scores, covariance):
    # The reference drops the first qualifying state and assigns zero to
    # grid points with singular covariance for the remaining states.
    contrast = scores[:, 1:]
    if not contrast.shape[1]:
        return np.zeros(len(scores))
    block = covariance[1:, 1:]
    try:
        inverse = np.linalg.inv(block)
    except np.linalg.LinAlgError:
        return np.zeros(len(scores))
    if np.linalg.norm(block, 1) * np.linalg.norm(inverse, 1) >= 1 / np.finfo(float).eps:
        return np.zeros(len(scores))
    return np.einsum("ij,jk,ik->i", contrast, inverse, contrast)


def markov_test(
    records: MultiStateRecords,
    transition: int,
    grid: ArrayLike,
    *,
    n_bootstrap: int | None = None,
    distribution: str = "poisson",
    random_state: int | np.random.Generator | None = None,
    multipliers: ArrayLike | None = None,
) -> MarkovTestResult:
    """Test a zero-based transition with standard, unadjusted log-rank weights.

    At each grid time subjects are classified by their state immediately
    before that time. Only subsequent events enter the trace. The default
    uses 1000 centered Poisson wild bootstrap draws. Supplied multipliers
    are arranged as (replicates, events), with event columns ordered by
    subject ID then state entry time. Set n_bootstrap=0 for observed traces.
    """
    episodes = _episodes(records)
    if (
        not isinstance(transition, Integral) or isinstance(transition, (bool, np.bool_))
        or not 0 <= transition < len(records.transitions)
    ):
        raise ValueError("transition must be a zero-based index into records.transitions")
    grid = _numeric_vector(grid, "grid")
    if grid[0] < 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be nonnegative and strictly increasing")
    origin, destination = map(int, records.transitions[transition])
    states = _qualifying_states(records.transitions, records.n_states, origin)
    at_risk = episodes.origin == origin
    observed = episodes.outcome == transition
    event_subject, event_exit = episodes.subject[observed], episodes.stop[observed]
    if not len(event_subject):
        raise ValueError("the selected transition must have at least one observed event")
    samples = _bootstrap_multipliers(multipliers, n_bootstrap, distribution, random_state, len(event_subject))

    times, time_index, deaths = np.unique(event_exit, return_inverse=True, return_counts=True)
    first = np.searchsorted(times, episodes.start[at_risk], side="right")
    after = np.searchsorted(times, episodes.stop[at_risk], side="right")
    size = len(times) + 1
    risk = np.cumsum(np.bincount(first, minlength=size) - np.bincount(after, minlength=size))[:-1]
    risk_subject = episodes.subject[at_risk]
    subject_count = len(np.unique(risk_subject))
    event_counts = len(event_exit) - np.searchsorted(np.sort(event_exit), grid, side="right")
    k = len(states)
    covariance = np.zeros((len(grid), k, k))
    raw_score = np.zeros((len(grid), k))
    all_z = np.zeros((len(samples) + 1, len(grid), k))
    all_chi = np.zeros((len(samples) + 1, len(grid)))

    for index, landmark in enumerate(grid):
        if not event_counts[index]:
            continue
        occupied = (episodes.start < landmark) & (landmark <= episodes.stop)
        occupancy = np.full(len(episodes.labels), -1, dtype=np.intp)
        occupancy[episodes.subject[occupied]] = episodes.origin[occupied]
        group = occupancy[risk_subject, None] == states
        entered = np.zeros((size, k), dtype=np.intp)
        np.add.at(entered, first, group)
        np.add.at(entered, after, -group.astype(np.intp))
        share = np.cumsum(entered, axis=0)[:-1] / risk[:, None]
        time_after = times > landmark
        after_share = share[time_after]
        after_deaths = deaths[time_after]
        expected = after_deaths @ after_share
        cov = np.diag(expected) - (after_share.T * after_deaths) @ after_share
        covariance[index] = cov

        event_after = event_exit > landmark
        observed_group = occupancy[event_subject[event_after], None] == states
        contribution = observed_group - share[time_index[event_after]]
        score = contribution.sum(axis=0)
        simulated = samples[:, event_after] @ contribution
        raw_score[index] = score
        combined = np.vstack((score, simulated))
        variance = np.diag(cov)
        all_z[:, index] = combined / np.sqrt(np.where(variance > 0, variance, 1))
        all_chi[:, index] = _joint_trace(combined, cov)

    state_statistics = np.abs(all_z).mean(axis=1)
    overall_statistics = all_chi.mean(axis=1)
    if len(samples):
        state_pvalues = (state_statistics[1:] > state_statistics[0]).mean(axis=0)
        overall_pvalue = float((overall_statistics[1:] > overall_statistics[0]).mean())
        quantiles = np.quantile(all_z[1:], [0.025, 0.975], axis=0)
    else:
        state_pvalues = overall_pvalue = quantiles = None
    return MarkovTestResult(
        int(transition), origin, destination, states, grid, subject_count, event_counts,
        episodes.labels[event_subject], event_exit, raw_score, all_z[0], covariance, all_chi[0],
        state_statistics[0], float(overall_statistics[0]), all_z[1:], all_chi[1:],
        state_statistics[1:], overall_statistics[1:], state_pvalues, overall_pvalue, quantiles,
    )
