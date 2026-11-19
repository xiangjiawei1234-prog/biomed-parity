"""Breslow cumulative hazards from transition-stratified counting processes."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cox import _fit_likelihood
from ._survival_inputs import _fix_near_ties
from .mstate_prep import MultiStateRecords
from .multistate import TransitionProbabilities, probtrans


@dataclass(frozen=True)
class TransitionHazards:
    """Cumulative hazard and covariance arrays on one absolute time grid."""

    times: NDArray[np.float64]
    transitions: NDArray[np.intp]
    n_states: int
    cumulative_hazards: NDArray[np.float64]
    cumulative_covariance: NDArray[np.float64] | None
    coefficients: NDArray[np.float64]
    coefficient_covariance: NDArray[np.float64]
    log_likelihood: float
    vartype: str

    def predict(self, *, predt=0, direction="forward", covariance=False) -> TransitionProbabilities:
        """Apply Aalen–Johansen using the fitted edge grid and variance method."""
        return probtrans(
            self.times, self.transitions, self.cumulative_hazards,
            predt=predt, direction=direction, n_states=self.n_states,
            cumulative_covariance=self.cumulative_covariance, method=self.vartype,
            covariance=covariance,
        )


@dataclass(frozen=True)
class _BreslowRiskSets:
    design: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    deaths: np.ndarray
    event_design: np.ndarray

    @classmethod
    def build(cls, start, stop, event, design, times):
        first = np.searchsorted(times, start, side="right")
        after = np.searchsorted(times, stop, side="right")
        present = first < after
        death_indices = np.searchsorted(times, stop[event])
        deaths = np.bincount(death_indices, minlength=len(times))
        event_design = np.zeros((len(times), design.shape[1]))
        np.add.at(event_design, death_indices, design[event])
        return cls(design[present], first[present], after[present], deaths, event_design)

    def risk_cumulative(self, weighted):
        size = len(self.deaths) + 1
        added = np.bincount(self.entry, weights=weighted, minlength=size)
        removed = np.bincount(self.exit, weights=weighted, minlength=size)
        return np.cumsum(added - removed)[:-1]

    def moments(self, beta):
        predictor = self.design @ beta
        shift = predictor.max()
        risk = np.exp(predictor - shift)
        risk_set = self.risk_cumulative(risk)
        if np.any(risk_set <= 0):
            raise ValueError("Cox risk sets do not have a stable positive total")
        mean = np.empty((len(risk_set), len(beta)))
        for column, values in enumerate(self.design.T):
            mean[:, column] = self.risk_cumulative(risk * values) / risk_set
        return risk_set, mean, risk, shift

    def evaluate(self, beta, _ties):
        risk_set, mean, risk, shift = self.moments(beta)
        terms = self.event_design @ beta - self.deaths * (np.log(risk_set) + shift)
        gradient = self.event_design.sum(axis=0) - self.deaths @ mean
        information = -(mean.T * self.deaths) @ mean
        inverse_risk = self.deaths / risk_set
        for column, x in enumerate(self.design.T):
            for preceding in range(column + 1):
                y = self.design[:, preceding]
                total = self.risk_cumulative(risk * x * y) @ inverse_risk
                information[column, preceding] += total
                if preceding != column:
                    information[preceding, column] += total
        return terms.sum(), gradient, information


def _numbers(values, name, dimensions):
    array = np.asarray(values)
    if array.ndim != dimensions or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric array with {dimensions} dimensions")
    return array.astype(float)


def _counting_data(t_start, t_stop, status, transition, transitions, n_states, timefix):
    if isinstance(t_start, MultiStateRecords):
        if any(value is not None for value in (t_stop, status, transition, transitions)):
            raise ValueError("supply the records object or individual counting-process arrays")
        records = t_start
        t_start, t_stop, status, transition, transitions = (
            records.t_start, records.t_stop, records.status, records.transition, records.transitions,
        )
        n_states = records.n_states if n_states is None else n_states
    start = _numbers(t_start, "t_start", 1)
    stop = _numbers(t_stop, "t_stop", 1)
    event = _numbers(status, "status", 1)
    edge = _numbers(transition, "transition", 1)
    pairs = np.asarray(transitions)
    if not start.shape == stop.shape == event.shape == edge.shape:
        raise ValueError("all counting-process columns must have the same length")
    if (
        pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
        or pairs.dtype.kind not in "iu" or np.any(pairs < 0)
        or np.any(pairs[:, 0] == pairs[:, 1])
        or len(np.unique(pairs, axis=0)) != len(pairs)
    ):
        raise ValueError("transitions must contain distinct nonnegative integer state pairs")
    if np.any(~np.isfinite(edge) | (edge != np.floor(edge)) | (edge < 0) | (edge >= len(pairs))):
        raise ValueError("transition must contain edge indices from transitions")
    if np.any((event != 0) & (event != 1)):
        raise ValueError("status must contain only 0 or 1")
    event = event.astype(bool)
    if (
        np.any(~np.isfinite(start) | (start < 0) | np.isnan(stop) | (stop <= start))
        or np.any(event & ~np.isfinite(stop))
    ):
        raise ValueError("intervals require 0 <= t_start < t_stop and a finite event time")
    minimum_states = int(pairs.max()) + 1
    if n_states is None:
        n_states = minimum_states
    if not isinstance(n_states, Integral) or n_states < minimum_states:
        raise ValueError("n_states must include all transition endpoints")
    if timefix and len(start):
        finite = np.isfinite(stop)
        snap = _fix_near_ties(np.r_[start, stop[finite]])
        start = snap[:len(start)]
        stop[finite] = snap[len(start):]
        if np.any(stop <= start):
            raise ValueError("near-tie correction created an interval of zero length")
    return start, stop, event, edge.astype(np.intp), pairs.astype(np.intp, copy=True), int(n_states)


def _fit_design(covariates, new_covariates, rows, edges):
    matrix = np.empty((rows, 0)) if covariates is None else np.asarray(covariates)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    matrix = _numbers(matrix, "covariates", 2)
    if len(matrix) != rows or not np.isfinite(matrix).all():
        raise ValueError("covariates must contain finite numeric columns aligned with intervals")
    dimensions = matrix.shape[1]
    targets = np.zeros((edges, dimensions)) if new_covariates is None else np.asarray(new_covariates)
    if targets.shape == (dimensions,):
        targets = np.broadcast_to(targets, (edges, dimensions))
    targets = _numbers(targets, "new_covariates", 2)
    if targets.shape != (edges, dimensions) or not np.isfinite(targets).all():
        raise ValueError("new_covariates must supply a finite covariate row for each edge")
    center = matrix.mean(axis=0) if rows else np.zeros(dimensions)
    scale = matrix.std(axis=0) if rows else np.ones(dimensions)
    scale = np.where(scale > 0, scale, 1)
    return (matrix - center) / scale, (targets - center) / scale, scale


def _fixed_parameters(coefficients, coefficient_covariance, scale):
    size = len(scale)
    beta = _numbers(coefficients, "coefficients", 1)
    if beta.shape != (size,) or not np.isfinite(beta).all():
        raise ValueError("coefficients must supply one finite value per covariate column")
    covariance = (
        np.zeros((size, size)) if coefficient_covariance is None
        else _numbers(coefficient_covariance, "coefficient_covariance", 2)
    )
    if (
        covariance.shape != (size, size) or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
        or (size and np.linalg.eigvalsh(covariance).min() < -1e-12)
    ):
        raise ValueError("coefficient_covariance must be finite, symmetric and positive semidefinite")
    return beta * scale, covariance * np.outer(scale, scale)


def _greenwood_covariance(times, begin, end, edges, transition, death_count):
    covariance = np.zeros((len(times), len(edges), len(edges)))
    for origin in np.unique(edges[:, 0]):
        outgoing = np.flatnonzero(edges[:, 0] == origin)
        indices = np.flatnonzero(death_count[:, outgoing].sum(axis=1))
        if not len(indices):
            continue
        at_risk = []
        for edge in outgoing:
            rows = transition == edge
            first = np.searchsorted(np.sort(begin[rows]), times[indices], side="left")
            last = np.searchsorted(np.sort(end[rows]), times[indices], side="left")
            at_risk.append(first - last)
        counts = at_risk[0]
        deaths = death_count[np.ix_(indices, outgoing)]
        if (
            not np.all(np.asarray(at_risk) == counts)
            or np.any(deaths.sum(axis=1) > counts)
            or np.any(counts <= 0)
        ):
            raise ValueError("Greenwood covariance requires a common risk set for all edges from a state")
        relative = deaths / counts[:, None]
        increments = -relative[:, :, None] * relative[:, None, :]
        diagonal = np.arange(len(outgoing))
        increments[:, diagonal, diagonal] += relative
        covariance[np.ix_(indices, outgoing, outgoing)] = increments / counts[:, None, None]
    return np.cumsum(covariance, axis=0)


def msfit(
    t_start: MultiStateRecords | ArrayLike,
    t_stop: ArrayLike | None = None,
    status: ArrayLike | None = None,
    transition: ArrayLike | None = None,
    transitions: ArrayLike | None = None,
    *,
    covariates: ArrayLike | None = None,
    new_covariates: ArrayLike | None = None,
    coefficients: ArrayLike | None = None,
    coefficient_covariance: ArrayLike | None = None,
    variance: bool = True,
    vartype: str = "aalen",
    n_states: int | None = None,
    timefix: bool = True,
) -> TransitionHazards:
    """Fit separate transition baselines in an unweighted Breslow Cox model.

    Give multi-state records from msprep or individual counting-process
    arrays. The risk set uses (t_start, t_stop]. Covariates are interval-
    aligned and coefficients are shared by the transition strata. A new
    numeric design row can be supplied per edge, with default all zeros.
    Greenwood variance applies only without covariates and with a common
    risk set for edges from each origin. Aalen variance uses the model-
    based coefficient covariance if it was fitted or explicitly supplied.
    """
    if vartype not in ("aalen", "greenwood"):
        raise ValueError("vartype must be 'aalen' or 'greenwood'")
    if not isinstance(variance, (bool, np.bool_)) or not isinstance(timefix, (bool, np.bool_)):
        raise ValueError("variance and timefix must be boolean")
    start, stop, event, transition, edges, n_states = _counting_data(
        t_start, t_stop, status, transition, transitions, n_states, timefix,
    )
    design, targets, scale = _fit_design(covariates, new_covariates, len(start), len(edges))
    dimensions = len(scale)
    if variance and vartype == "greenwood" and dimensions:
        raise ValueError("Greenwood variance is defined here only without covariates")
    if coefficients is None and coefficient_covariance is not None:
        raise ValueError("explicit coefficient_covariance requires explicit coefficients")

    finite_stops = stop[np.isfinite(stop)]
    last_followup = finite_stops.max(initial=0)
    times = np.unique(np.r_[stop[event], last_followup])
    strata = []
    for edge in range(len(edges)):
        rows = transition == edge
        failures = np.unique(stop[rows & event])
        if len(failures):
            stratum = _BreslowRiskSets.build(start[rows], stop[rows], event[rows], design[rows], failures)
            strata.append((edge, np.searchsorted(times, failures), stratum))

    if coefficients is not None:
        beta, beta_covariance = _fixed_parameters(coefficients, coefficient_covariance, scale)
    elif dimensions:
        if not strata:
            raise ValueError("cannot fit transition coefficients without events")
        beta, _, _, beta_covariance = _fit_likelihood(
            [stratum for _, _, stratum in strata], dimensions, "breslow",
        )
    else:
        beta, beta_covariance = np.empty(0), np.empty((0, 0))

    shape = (len(times), len(edges))
    hazards = np.zeros(shape)
    use_aalen = variance and vartype == "aalen"
    use_greenwood = variance and vartype == "greenwood"
    direct_variance = np.zeros(shape) if use_aalen else None
    death_count = np.zeros(shape, dtype=np.intp) if use_greenwood else None
    gradients = np.zeros((*shape, dimensions)) if use_aalen and dimensions else None
    log_likelihood = 0.0
    for edge, indices, stratum in strata:
        risk, mean, _, shift = stratum.moments(beta)
        d = stratum.deaths
        log_likelihood += (stratum.event_design @ beta - d * (np.log(risk) + shift)).sum()
        with np.errstate(over="ignore", invalid="ignore"):
            jump_per_death = np.exp(targets[edge] @ beta - shift - np.log(risk))
            hazards[indices, edge] = d * jump_per_death
            if direct_variance is not None:
                direct_variance[indices, edge] = d * jump_per_death**2
            if gradients is not None:
                gradients[indices, edge] = hazards[indices, edge, None] * (targets[edge] - mean)
        if death_count is not None:
            death_count[indices, edge] = d

    covariance = None
    if use_greenwood:
        covariance = _greenwood_covariance(times, start, stop, edges, transition, death_count)
    with np.errstate(over="ignore", invalid="ignore"):
        if use_aalen:
            covariance = np.zeros((*shape, len(edges)))
            if gradients is not None:
                gradient = np.cumsum(gradients, axis=0)
                covariance = (gradient @ beta_covariance) @ gradient.swapaxes(1, 2)
            diagonal = np.arange(len(edges))
            covariance[:, diagonal, diagonal] += np.cumsum(direct_variance, axis=0)
        cumulative = np.cumsum(hazards, axis=0)
    if not np.isfinite(cumulative).all() or (covariance is not None and not np.isfinite(covariance).all()):
        raise ValueError("new_covariates produce a nonfinite cumulative hazard or covariance")

    return TransitionHazards(
        times, edges, n_states, cumulative, covariance,
        beta / scale, beta_covariance / np.outer(scale, scale), float(log_likelihood), vartype,
    )
