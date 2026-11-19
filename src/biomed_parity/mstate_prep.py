"""Prepare counting-process records for progressive multi-state models."""

from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from numbers import Real
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _encode_labels


class MsprepWarning(UserWarning):
    """A wide record has ambiguous or conflicting outgoing transition times."""


@dataclass(frozen=True)
class MultiStateRecords:
    """Transition-risk rows sorted by subject ID, entry, origin and destination.

    States, transition numbers and input subject indices are zero-based.
    """

    id: NDArray
    subject_index: NDArray[np.intp]
    from_state: NDArray[np.intp]
    to_state: NDArray[np.intp]
    transition: NDArray[np.intp]
    t_start: NDArray[np.float64]
    t_stop: NDArray[np.float64]
    status: NDArray[np.intp]
    transitions: NDArray[np.intp]
    n_states: int
    covariates: dict[str, NDArray]

    @property
    def time(self) -> NDArray[np.float64]:
        """Duration of each row's interval."""
        return self.t_stop - self.t_start

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """R-style transition matrix with 1-based edge numbers and NaN elsewhere."""
        matrix = np.full((self.n_states, self.n_states), np.nan)
        matrix[tuple(self.transitions.T)] = np.arange(1, len(self.transitions) + 1)
        return matrix


def _numeric_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" and (
        array.dtype.kind != "O"
        or not all(item is None or isinstance(item, (Real, np.bool_)) for item in array.flat)
    ):
        raise ValueError(f"{name} must contain real numbers or missing values")
    return array.astype(float)


def _column(data, name):
    try:
        return np.asarray(data[name])
    except KeyError as exc:
        raise ValueError(f"column {name!r} is absent from data") from exc


def _wide_matrix(values, name, data):
    values = np.asarray(values)
    if values.ndim == 1 and data is not None and all(
        item is None or isinstance(item, str) for item in values
    ):
        named = [_column(data, item) if item is not None else None for item in values]
        present = [column for column in named if column is not None]
        if not present or present[0].ndim != 1:
            raise ValueError(f"{name} must select at least one one-dimensional data column")
        shape = present[0].shape
        if any(column.shape != shape for column in present):
            raise ValueError(f"{name} columns must have the same length")
        values = np.column_stack([
            np.full(shape, np.nan) if column is None else column for column in named
        ])
    result = _numeric_array(values, name)
    if result.ndim != 2 or not result.size:
        raise ValueError(f"{name} must have shape (subjects, states)")
    return result


def _edges_and_order(transitions, transition_matrix, n_states):
    if (transitions is None) == (transition_matrix is None):
        raise ValueError("supply exactly one of transitions or transition_matrix")
    if transition_matrix is not None:
        matrix = _numeric_array(transition_matrix, "transition_matrix")
        if matrix.shape != (n_states, n_states):
            raise ValueError("transition_matrix must be square and match the state columns")
        edges = np.argwhere(~np.isnan(matrix))
        labels = matrix[tuple(edges.T)]
        if not np.array_equal(np.sort(labels), np.arange(1, len(edges) + 1)):
            raise ValueError("transition_matrix must label edges consecutively starting at 1")
        edges = edges[np.argsort(labels)]
    else:
        edges = np.asarray(transitions)
        if edges.shape == (0,):
            edges = np.empty((0, 2), dtype=np.intp)
        if (
            edges.ndim != 2 or edges.shape[1] != 2
            or edges.dtype.kind not in "iu"
            or np.any((edges < 0) | (edges >= n_states))
        ):
            raise ValueError("transitions must contain integer state pairs within the state columns")
        edges = edges.astype(np.intp, copy=True)
    if len(np.unique(edges, axis=0)) != len(edges) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("transitions must contain distinct edges between different states")

    destinations = edges[:, 1]
    incoming = np.bincount(destinations, minlength=n_states)
    outgoing = [[] for _ in range(n_states)]
    for edge, origin in enumerate(edges[:, 0]):
        outgoing[origin].append(edge)
    outgoing = [np.asarray(sorted(indices, key=destinations.__getitem__), dtype=np.intp)
                for indices in outgoing]

    ready = np.flatnonzero(incoming == 0).tolist()
    is_destination = incoming > 0
    heapify(ready)
    order = []
    while ready:
        origin = heappop(ready)
        order.append(origin)
        for destination in destinations[outgoing[origin]]:
            incoming[destination] -= 1
            if incoming[destination] == 0:
                heappush(ready, int(destination))
    if len(order) != n_states:
        raise ValueError("the transition graph must be acyclic")
    return edges, outgoing, order, is_destination


def _start_values(values, size, name):
    values = _numeric_array(values, name)
    if values.ndim == 0 or values.shape == (1,):
        values = np.full(size, values.item())
    if values.shape != (size,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must contain finite nonnegative values for each subject")
    return values.copy()


def _covariates(keep, data, size):
    if keep is None:
        return {}
    if isinstance(keep, Mapping):
        columns = {name: np.asarray(values) for name, values in keep.items()}
    else:
        values = np.asarray(keep)
        if values.ndim <= 1 and data is not None and values.dtype.kind in "OUS" and all(
            isinstance(name, str) for name in np.atleast_1d(values)
        ):
            columns = {name: _column(data, name) for name in np.atleast_1d(values)}
        else:
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or len(values) != size:
                raise ValueError("keep must have one row per subject")
            columns = {f"keep{index + 1}": column for index, column in enumerate(values.T)}
    if any(not isinstance(name, str) or values.shape != (size,) for name, values in columns.items()):
        raise ValueError("keep must contain named one-dimensional columns with one row per subject")
    return columns


def _select_outcome(next_time, next_status, entry, origin, ids):
    eligible = np.where(next_time >= entry[:, None], next_time, np.inf)
    event_time = np.where(next_status == 1, eligible, np.inf)
    best = np.argmin(event_time, axis=1)
    first_event = event_time[np.arange(len(best)), best]
    observed = np.isfinite(first_event)
    first_record = np.min(eligible, axis=1)

    earlier_censor = observed & (first_record < first_event)
    if earlier_censor.any():
        subjects = ", ".join(map(str, ids[earlier_censor]))
        warn(
            f"In state {origin}, a censored time precedes an observed transition for subject IDs "
            f"{subjects}. The observed transition is used.",
            MsprepWarning,
            stacklevel=3,
        )
    ties = observed & (np.sum(event_time == first_event[:, None], axis=1) > 1)
    if ties.any():
        subjects = ", ".join(map(str, ids[ties]))
        warn(
            f"In state {origin}, observed transitions are simultaneous for subject IDs "
            f"{subjects}. The smallest destination state is used.",
            MsprepWarning,
            stacklevel=3,
        )

    stop = np.where(observed, first_event, first_record)
    return best, observed, stop


def msprep(
    time: ArrayLike,
    status: ArrayLike,
    transitions: ArrayLike | None = None,
    *,
    transition_matrix: ArrayLike | None = None,
    start_state: ArrayLike = 0,
    start_time: ArrayLike = 0,
    id: ArrayLike | None = None,
    id_order: ArrayLike | None = None,
    keep: Mapping[str, ArrayLike] | ArrayLike | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> MultiStateRecords:
    """Convert wide state times and event indicators into transition-risk rows.

    Columns of time and status correspond to states. With data, they may
    also be specified by column names with None for unused source columns.
    Supply zero-based edge pairs in transition-number order or an R-style
    transition_matrix. Scalar starting states and entry times broadcast.
    The default IDs are 1 through the number of input subjects.
    """
    time = _wide_matrix(time, "time", data)
    status = _wide_matrix(status, "status", data)
    if time.shape != status.shape:
        raise ValueError("time and status must have the same shape")
    size, n_states = time.shape
    edges, outgoing, order, is_destination = _edges_and_order(
        transitions, transition_matrix, n_states,
    )
    relevant_times = time[:, is_destination]
    relevant_status = status[:, is_destination]
    if not (np.isfinite(relevant_times).all() and np.all(relevant_times >= 0)):
        raise ValueError("time must be finite and nonnegative in all destination states")
    if not np.all((relevant_status == 0) | (relevant_status == 1)):
        raise ValueError("status must be 0 or 1 in all destination states")

    states = _start_values(start_state, size, "start_state")
    if np.any((states != np.floor(states)) | (states >= n_states)):
        raise ValueError("start_state must contain integer states within the state columns")
    states = states.astype(np.intp)
    entries = _start_values(start_time, size, "start_time")
    if id is None:
        id = np.arange(1, size + 1)
    if isinstance(id, str) and data is not None:
        id = _column(data, id)
    ids = np.asarray(id, dtype=object)
    levels, id_codes = _encode_labels(ids, id_order, size, "id")
    if len(levels) != size:
        raise ValueError("id must identify each subject uniquely")
    columns = _covariates(keep, data, size)

    subjects = []
    edge_rows = []
    starts = []
    stops = []
    observations = []
    for origin in order:
        indices = outgoing[origin]
        rows = np.flatnonzero(states == origin)
        if not len(indices) or not len(rows):
            continue
        destinations = edges[indices, 1]
        subscript = np.ix_(rows, destinations)
        entry = entries[rows]
        best, observed, stop = _select_outcome(time[subscript], status[subscript], entry, origin, ids[rows])

        width = len(destinations)
        subjects.append(np.repeat(rows, width))
        edge_rows.append(np.tile(indices, len(rows)))
        starts.append(np.repeat(entry, width))
        stops.append(np.repeat(stop, width))
        observations.append((observed[:, None] & (best[:, None] == np.arange(width))).ravel())

        advanced = rows[observed]
        states[advanced] = destinations[best[observed]]
        entries[advanced] = stop[observed]

    if subjects:
        subject = np.concatenate(subjects)
        transition = np.concatenate(edge_rows)
        t_start = np.concatenate(starts)
        t_stop = np.concatenate(stops)
        event = np.concatenate(observations).astype(np.intp)
        origin, destination = edges[transition].T
        perm = np.lexsort((destination, origin, t_start, id_codes[subject]))
        subject, transition, t_start, t_stop, event = (
            values[perm] for values in (subject, transition, t_start, t_stop, event)
        )
    else:
        subject = transition = event = np.empty(0, dtype=np.intp)
        t_start = t_stop = np.empty(0)
    origin, destination = edges[transition].T
    return MultiStateRecords(
        ids[subject], subject, origin, destination, transition, t_start, t_stop, event,
        edges, n_states, {name: values[subject] for name, values in columns.items()},
    )
