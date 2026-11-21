"""Log-rank tests of the Markov assumption for multi-state transition data."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cox import _fit_likelihood
from ._survival_inputs import _encode_labels, _fix_near_ties, _numeric_vector
from .mstate_fit import _counting_data
from .mstate_prep import MultiStateRecords


@dataclass(frozen=True)
class MarkovCoxFit:
    """Efron partial likelihood for the selected transition."""

    covariates: tuple[str, ...]
    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    log_likelihood: float
    null_log_likelihood: float


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
    cox_fit: MarkovCoxFit | None = None


@dataclass(frozen=True)
class _Episodes:
    labels: NDArray
    subject: NDArray[np.intp]
    origin: NDArray[np.intp]
    start: NDArray[np.float64]
    stop: NDArray[np.float64]
    outcome: NDArray[np.intp]


def _episodes(records, *, warn_nonpositive=True):
    if not isinstance(records, MultiStateRecords):
        raise TypeError("records must be a MultiStateRecords object")
    drop = records.t_stop <= records.t_start
    if warn_nonpositive and drop.any():
        warn(f"removed {drop.sum()} rows with nonpositive interval duration", stacklevel=3)
    keep = ~drop
    start, stop, event, edge_index, edges, _ = _counting_data(
        records.t_start[keep], records.t_stop[keep], records.status[keep],
        records.transition[keep], records.transitions, records.n_states, timefix=False,
    )
    labels, subjects = _encode_labels(records.id[keep], None, len(start), "id")
    origin = edges[edge_index, 0]
    if not len(subjects):
        return _Episodes(np.asarray(labels), subjects, origin, start, stop, edge_index)
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


def _markov_grid(records, transition, grid):
    if (
        not isinstance(transition, Integral) or isinstance(transition, (bool, np.bool_))
        or not 0 <= transition < len(records.transitions)
    ):
        raise ValueError("transition must be a zero-based index into records.transitions")
    grid = _numeric_vector(grid, "grid")
    if grid[0] < 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be nonnegative and strictly increasing")
    return grid


def _optimal_weights(records, episodes, transition, grid):
    # The R weighting functions count all observed origins and future target
    # events in the original records, including any nonpositive-duration rows.
    states = np.unique(records.from_state)
    if not len(states):
        raise ValueError("records contain no state episodes")
    counts = np.empty((len(grid), len(states)), dtype=np.intp)
    for column, state in enumerate(states):
        rows = episodes.origin == state
        entered = np.searchsorted(np.sort(episodes.start[rows]), grid, side="right")
        exited = np.searchsorted(np.sort(episodes.stop[rows]), grid, side="right")
        counts[:, column] = entered - exited
    total = counts.sum(axis=1).astype(float)
    event_exit = np.sort(records.t_stop[(records.transition == transition) & (records.status == 1)])
    events_after = len(event_exit) - np.searchsorted(event_exit, grid, side="right")
    numerator = (total[:, None] - counts) * counts * events_after[:, None]
    state_weights = np.divide(
        np.sqrt(numerator), total[:, None], out=np.zeros_like(numerator), where=total[:, None] > 0,
    )
    overall_weights = np.divide(
        numerator.max(axis=1), total**2, out=np.zeros_like(total), where=total > 0,
    )
    return states, state_weights, overall_weights


def _grid_widths(grid, min_time):
    minimum = _numeric_vector([min_time], "min_time")[0]
    if minimum > grid[0]:
        raise ValueError("min_time must be at or before the first grid point")
    with np.errstate(over="ignore"):
        widths = np.diff(grid, prepend=minimum)
    if not np.isfinite(widths).all():
        raise ValueError("grid intervals from min_time must have finite widths")
    return widths


def optimal_weights_matrix(
    records: MultiStateRecords, transition: int, grid: ArrayLike,
) -> NDArray[np.float64]:
    """Return the raw optimal state weights in mstate's `weights` attribute.

    Rows correspond to grid points. Columns correspond to the sorted state
    indices observed as episode origins, including states outside the test's
    qualifying set. The matrix excludes the grid interval widths.
    """
    episodes = _episodes(records, warn_nonpositive=False)
    grid = _markov_grid(records, transition, grid)
    return _optimal_weights(records, episodes, transition, grid)[1]


def optimal_weights_multiple(
    records: MultiStateRecords, transition: int, grid: ArrayLike, *, min_time: float = 0,
) -> NDArray[np.float64]:
    """Return the optimal integration weights for the joint chi-squared trace."""
    episodes = _episodes(records, warn_nonpositive=False)
    grid = _markov_grid(records, transition, grid)
    return _optimal_weights(records, episodes, transition, grid)[2] * _grid_widths(grid, min_time)


def _summary_weights(weights, name, grid, states=None):
    if weights is None:
        return None
    array = np.asarray(weights)
    shape = (len(grid),)
    if states is not None:
        if array.shape == shape:
            array = array[:, None]
        elif array.shape != (len(grid), len(states)):
            raise ValueError(f"{name} must have shape (grid,) or (grid, qualifying_states)")
    elif array.shape != shape:
        raise ValueError(f"{name} must have one weight per grid point")
    array = _numeric_vector(array.ravel(), name).reshape(array.shape)
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _weighted_trace_mean(trace, weights):
    if weights is None:
        return trace.mean(axis=1)
    scale = weights.max(axis=0)
    scaled = np.divide(weights, scale, out=np.zeros_like(weights), where=scale > 0)
    total = scaled.sum(axis=0)
    weighted = (trace * scaled).sum(axis=1)
    return np.divide(weighted, total, out=np.full_like(weighted, np.nan), where=total > 0)


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


@dataclass(frozen=True)
class _RiskIntervals:
    first: NDArray[np.intp]
    after: NDArray[np.intp]
    size: int

    @classmethod
    def build(cls, times, start, stop):
        return cls(
            np.searchsorted(times, start, side="right"),
            np.searchsorted(times, stop, side="right"),
            len(times) + 1,
        )

    def total(self, values):
        change = np.zeros((self.size, *values.shape[1:]), dtype=values.dtype)
        np.add.at(change, self.first, values)
        np.subtract.at(change, self.after, values)
        return np.cumsum(change, axis=0)[:-1]


@dataclass(frozen=True)
class _EfronRiskSets:
    intervals: _RiskIntervals
    design: NDArray[np.float64]
    offset: NDArray[np.float64]
    event: NDArray[np.bool_]
    event_indices: NDArray[np.intp]
    ties: NDArray[np.intp]
    fractions: NDArray[np.float64]
    moments: NDArray[np.float64]

    @classmethod
    def build(cls, start, stop, event, design, offset):
        finite = np.isfinite(stop)
        snapped = _fix_near_ties(np.r_[start, stop[finite]])
        start = snapped[:len(start)]
        stop = stop.copy()
        stop[finite] = snapped[len(start):]
        if np.any(stop <= start):
            raise ValueError("near-tie correction created a Cox interval of zero length")
        times, event_indices, deaths = np.unique(stop[event], return_inverse=True, return_counts=True)
        ties = np.repeat(np.arange(len(times)), deaths)
        fractions = (np.arange(event.sum()) - np.repeat(np.cumsum(deaths) - deaths, deaths)) / deaths[ties]
        outer = (design[:, :, None] * design[:, None, :]).reshape(len(design), design.shape[1] ** 2)
        moments = np.column_stack((np.ones(len(design)), design, outer))
        return cls(_RiskIntervals.build(times, start, stop), design, offset, event,
                   event_indices, ties, fractions, moments)

    def evaluate(self, beta, _ties):
        eta = self.design @ beta + self.offset
        centered = eta - eta.max()
        values = self.moments * np.exp(centered[:, None])
        total = self.intervals.total(values)
        tied_events = np.zeros_like(total)
        np.add.at(tied_events, self.event_indices, values[self.event])
        remaining = total[self.ties] - self.fractions[:, None] * tied_events[self.ties]
        denominator = remaining[:, 0]
        if np.any(denominator <= 0) or not np.isfinite(denominator).all():
            raise ValueError("Cox risk sets do not have stable positive totals")
        width = len(beta)
        means = remaining[:, 1:width + 1] / denominator[:, None]
        second = (remaining[:, width + 1:] / denominator[:, None]).sum(axis=0).reshape(width, width)
        likelihood = centered[self.event].sum() - np.log(denominator).sum()
        gradient = self.design[self.event].sum(axis=0) - means.sum(axis=0)
        return likelihood, gradient, second - means.T @ means


def _fit_cox(records, risk_subject, start, stop, event, covariates, offset, cox_mode):
    if covariates is None:
        names = ()
    elif isinstance(covariates, str):
        names = (covariates,)
    elif isinstance(covariates, Sequence):
        names = tuple(covariates)
    else:
        raise ValueError("covariates must name one or more numeric record columns")
    if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("covariates must contain distinct names of numeric record columns")
    if offset is not None and not isinstance(offset, str):
        raise ValueError("offset must name a numeric record column")
    if not names and offset is None:
        return None, np.empty((len(start), 0)), np.zeros(len(start)), np.empty((0, 0))

    # msdata2etm takes the first retained row's covariates for each subject.
    rows = np.flatnonzero(records.t_start < records.t_stop)
    _, first = np.unique(records.id[rows], return_index=True)
    selected = rows[first[risk_subject]]

    def column(name):
        if name not in records.covariates:
            raise ValueError(f"Cox column {name!r} is absent from records.covariates")
        values = np.asarray(records.covariates[name])
        if values.shape != records.id.shape:
            raise ValueError(f"Cox column {name!r} must align with the record rows")
        return _numeric_vector(values[selected], name)

    raw_design = np.column_stack([column(name) for name in names]) if names else np.empty((len(start), 0))
    offset_values = np.zeros(len(start)) if offset is None else column(offset)
    offset_values = offset_values - (offset_values.min() * 0.5 + offset_values.max() * 0.5)
    scales = raw_design.std(axis=0)
    if np.any(scales == 0):
        raise ValueError("Cox covariates must be nonconstant")
    design = (raw_design - raw_design.mean(axis=0)) / scales
    if names and np.linalg.matrix_rank(design) < len(names):
        raise ValueError("Cox covariates must form a full-rank design")
    risk_sets = _EfronRiskSets.build(start, stop, event, design, offset_values)
    if names:
        beta, likelihood, null, inverse_information = _fit_likelihood([risk_sets], len(names), "efron")
    else:
        beta = np.empty(0)
        likelihood, _, inverse_information = risk_sets.evaluate(beta, "efron")
        null = likelihood
    fit = MarkovCoxFit(
        names, beta / scales, inverse_information / np.outer(scales, scales),
        float(likelihood), float(null),
    )
    predictor = design @ beta + (offset_values if cox_mode == "full" else 0)
    if cox_mode == "mstate" and len(names) > 1:
        return fit, raw_design, predictor, fit.covariance
    return fit, design, predictor, inverse_information


@dataclass(frozen=True)
class _CoxProjection:
    design: NDArray[np.float64]
    mean: NDArray[np.float64]
    inverse_information: NDArray[np.float64]
    variance: NDArray[np.float64]
    legacy_phases: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]

    @classmethod
    def build(cls, intervals, design, weights, risk, risk_event, time_index, inverse_information, cox_mode):
        mean = intervals.total(weights[:, None] * design) / risk[:, None]
        outer = design[:, :, None] * design[:, None, :]
        second = intervals.total(weights[:, None, None] * outer) / risk[:, None, None]
        variance = second - mean[:, :, None] * mean[:, None, :]
        legacy_phases = ()
        width = design.shape[1]
        if cox_mode == "mstate" and width > 1:
            # R recycles each length-p row through a column-major n-by-p
            # matrix. Event rows of the same phase use the same permutation.
            phases = np.flatnonzero(risk_event) % width
            offsets = len(design) * np.arange(width)
            legacy_phases = tuple(
                ((phase + offsets) % width, np.bincount(time_index[phases == phase], minlength=len(mean)))
                for phase in np.unique(phases)
            )
        return cls(design, mean, inverse_information, variance, legacy_phases)

    def at_landmark(self, intervals, weighted_group, weights, risk, share, deaths, time_after, eligible):
        cross = intervals.total(weighted_group[:, None, :] * self.design[:, :, None])
        centered = cross / risk[:, None, None] - self.mean[:, :, None] * share[:, None, :]
        late = np.einsum("t,tij->ij", deaths[time_after], centered[time_after])
        # Earlier event times enter H only for intervals ending after the landmark.
        if not time_after.all():
            early_weight = weights * eligible
            early_group = weighted_group * eligible[:, None]
            early_cross = intervals.total(early_group[:, None, :] * self.design[:, :, None])
            early_cross -= intervals.total(early_weight[:, None] * self.design)[:, :, None] * share[:, None, :]
            centered_mean = intervals.total(early_group) - intervals.total(early_weight)[:, None] * share
            early_cross -= self.mean[:, :, None] * centered_mean[:, None, :]
            early = np.einsum("t,tij->ij", deaths[~time_after] / risk[~time_after], early_cross[~time_after])
        else:
            early = 0
        projection = self.inverse_information @ (late + early)
        if self.legacy_phases:
            return projection, self._legacy_variance(projection, centered, share, time_after)
        information = np.einsum("t,tij->ij", deaths, self.variance)
        cross_term = late.T @ projection
        adjustment = projection.T @ information @ projection - cross_term - cross_term.T
        return projection, adjustment

    def _legacy_variance(self, projection, centered, share, time_after):
        adjustment = np.zeros((projection.shape[1], projection.shape[1]))
        for permutation, counts in self.legacy_phases:
            recycled = np.zeros_like(projection)
            np.add.at(recycled, permutation, projection)
            difference = self.mean @ (recycled - projection)
            after_counts = counts[time_after]
            variance = np.einsum("t,tij->ij", counts, self.variance)
            cross = np.einsum("t,tij->ij", after_counts, centered[time_after]).T @ recycled
            adjustment += (
                recycled.T @ variance @ recycled - cross - cross.T
                + difference.T @ (counts[:, None] * difference)
            )
            # R centers the second factor using the first state's mean.
            # Its upper triangle is mirrored to fill the covariance matrix.
            asymmetric = -(difference[time_after].T * after_counts) @ share[time_after]
            upper = np.triu(asymmetric - np.diag(asymmetric)[:, None], k=1)
            adjustment += upper + upper.T
        return adjustment


def markov_test(
    records: MultiStateRecords,
    transition: int,
    grid: ArrayLike,
    *,
    n_bootstrap: int | None = None,
    distribution: str = "poisson",
    random_state: int | np.random.Generator | None = None,
    multipliers: ArrayLike | None = None,
    state_weights: ArrayLike | Literal["optimal"] | None = None,
    overall_weights: ArrayLike | Literal["optimal"] | None = None,
    min_time: float = 0,
    covariates: str | Sequence[str] | None = None,
    offset: str | None = None,
    cox_mode: Literal["mstate", "full"] = "mstate",
) -> MarkovTestResult:
    """Test the Markov assumption for a zero-based transition.

    At each grid time subjects are classified by their state immediately
    before that time. Only subsequent events enter the trace. The default
    uses 1000 centered Poisson wild bootstrap draws. Supplied multipliers
    are arranged as (replicates, events), with event columns ordered by
    subject ID then state entry time. Set n_bootstrap=0 for observed traces.

    State and overall summaries default to arithmetic means. Use "optimal"
    for data-derived weights integrated over intervals starting at min_time.
    Explicit weights have one entry per grid point, with an optional column
    per qualifying state in state_weights. A zero total weight gives NaN.

    Covariates and offset name numeric columns in records.covariates.
    Covariates are fitted with the Efron partial likelihood. cox_mode="full"
    applies offsets to the test's risk weights and centers each covariance
    term by its own risk-set mean. The default matches mstate 0.3.3 with
    survival 3.8.6, including its offset and covariance conventions.
    """
    episodes = _episodes(records)
    if not len(episodes.labels):
        raise ValueError("records contain no positive-duration intervals")
    grid = _markov_grid(records, transition, grid)
    origin, destination = map(int, records.transitions[transition])
    states = _qualifying_states(records.transitions, records.n_states, origin)
    optimal_state = isinstance(state_weights, str) and state_weights == "optimal"
    optimal_overall = isinstance(overall_weights, str) and overall_weights == "optimal"
    if optimal_state or optimal_overall:
        widths = _grid_widths(grid, min_time)
        weight_states, raw_state, raw_overall = _optimal_weights(records, episodes, transition, grid)
        if optimal_state:
            state_weights = np.zeros((len(grid), len(states)))
            present = np.isin(states, weight_states)
            state_weights[:, present] = raw_state[:, np.searchsorted(weight_states, states[present])]
            state_weights *= widths[:, None]
        if optimal_overall:
            overall_weights = raw_overall * widths
    state_weights = _summary_weights(state_weights, "state_weights", grid, states)
    overall_weights = _summary_weights(overall_weights, "overall_weights", grid)
    if cox_mode not in ("mstate", "full"):
        raise ValueError("cox_mode must be 'mstate' or 'full'")
    at_risk = episodes.origin == origin
    observed = episodes.outcome == transition
    event_subject, event_exit = episodes.subject[observed], episodes.stop[observed]
    if not len(event_subject):
        raise ValueError("the selected transition must have at least one observed event")
    samples = _bootstrap_multipliers(multipliers, n_bootstrap, distribution, random_state, len(event_subject))

    times, time_index, deaths = np.unique(event_exit, return_inverse=True, return_counts=True)
    risk_subject = episodes.subject[at_risk]
    risk_start, risk_stop, risk_event = episodes.start[at_risk], episodes.stop[at_risk], observed[at_risk]
    cox_fit, design, predictor, inverse_information = _fit_cox(
        records, risk_subject, risk_start, risk_stop, risk_event, covariates, offset, cox_mode,
    )
    intervals = _RiskIntervals.build(times, risk_start, risk_stop)
    risk_weight = np.exp(predictor - predictor.max())
    risk = intervals.total(risk_weight)
    if np.any(risk <= 0) or not np.isfinite(risk).all():
        raise ValueError("Cox-adjusted trace risk sets do not have stable positive totals")
    cox_projection = None
    if design.shape[1]:
        cox_projection = _CoxProjection.build(
            intervals, design, risk_weight, risk, risk_event, time_index, inverse_information, cox_mode,
        )
        cox_bootstrap_score = samples @ (design[risk_event] - cox_projection.mean[time_index])

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
        weighted_group = risk_weight[:, None] * group
        share = intervals.total(weighted_group) / risk[:, None]
        time_after = times > landmark
        after_share = share[time_after]
        after_deaths = deaths[time_after]
        expected = after_deaths @ after_share
        cov = np.diag(expected) - (after_share.T * after_deaths) @ after_share

        event_after = event_exit > landmark
        observed_group = occupancy[event_subject[event_after], None] == states
        contribution = observed_group - share[time_index[event_after]]
        score = contribution.sum(axis=0)
        simulated = samples[:, event_after] @ contribution
        if cox_projection is not None:
            projection, adjustment = cox_projection.at_landmark(
                intervals, weighted_group, risk_weight, risk, share, deaths, time_after, risk_stop > landmark,
            )
            cov += adjustment
            simulated -= cox_bootstrap_score @ projection
        covariance[index] = cov
        raw_score[index] = score
        combined = np.vstack((score, simulated))
        variance = np.diag(cov)
        all_z[:, index] = combined / np.sqrt(np.where(variance > 0, variance, 1))
        all_chi[:, index] = _joint_trace(combined, cov)

    state_statistics = _weighted_trace_mean(np.abs(all_z), state_weights)
    overall_statistics = _weighted_trace_mean(all_chi, overall_weights)
    if len(samples):
        state_pvalues = (state_statistics[1:] > state_statistics[0]).mean(axis=0)
        state_pvalues[np.isnan(state_statistics[0])] = np.nan
        overall_pvalue = (
            np.nan if np.isnan(overall_statistics[0])
            else float((overall_statistics[1:] > overall_statistics[0]).mean())
        )
        quantiles = np.quantile(all_z[1:], [0.025, 0.975], axis=0)
    else:
        state_pvalues = overall_pvalue = quantiles = None
    return MarkovTestResult(
        int(transition), origin, destination, states, grid, subject_count, event_counts,
        episodes.labels[event_subject], event_exit, raw_score, all_z[0], covariance, all_chi[0],
        state_statistics[0], float(overall_statistics[0]), all_z[1:], all_chi[1:],
        state_statistics[1:], overall_statistics[1:], state_pvalues, overall_pvalue, quantiles, cox_fit,
    )
