"""Numerical nomogram axes for an explicit prefit regression design."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import prod
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rms_prediction import DataDistribution, prediction_grid


@dataclass(frozen=True)
class NomogramTerm:
    """One set of coefficients evaluated across a predictor and interaction profiles."""

    columns: Sequence[int]
    ticks: ArrayLike | None = None
    by: Mapping[str, ArrayLike | None] = field(default_factory=dict)


@dataclass(frozen=True)
class PointsAxis:
    predictor: str
    by: dict[str, object]
    ticks: NDArray
    effects: NDArray[np.float64]
    points: NDArray[np.float64]


@dataclass(frozen=True)
class OutcomeAxis:
    ticks: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    points: NDArray[np.float64]


@dataclass(frozen=True)
class Nomogram:
    predictor_axes: tuple[PointsAxis, ...]
    linear_axis: OutcomeAxis
    outcome_axes: dict[str, OutcomeAxis]
    points_per_unit: float
    zero_point_predictor: float
    maximum_total_points: float

    def to_linear_predictor(self, points: ArrayLike) -> NDArray[np.float64]:
        """Convert total nomogram points to the model's linear predictor."""
        return self.zero_point_predictor + np.asarray(points, dtype=float) / self.points_per_unit


def _numeric_vector(values, name):
    result = np.asarray(values)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "biuf"
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    return result.astype(float)


def _term_columns(terms, coefficients, intercept_column):
    if intercept_column is not None and (
        not isinstance(intercept_column, Integral) or isinstance(intercept_column, bool)
        or not 0 <= intercept_column < len(coefficients)
    ):
        raise ValueError("intercept_column must be an in-range index or None")
    groups = {}
    available = set(range(len(coefficients)))
    if intercept_column is not None:
        available.remove(intercept_column)
    for name, term in terms.items():
        if not isinstance(term, NomogramTerm):
            raise ValueError("each term must be a NomogramTerm")
        columns = np.asarray(term.columns)
        if (columns.ndim != 1 or not len(columns) or columns.dtype.kind not in "iu"
                or len(set(columns)) != len(columns) or not set(columns) <= available):
            raise ValueError("nomogram terms must partition the non-intercept design columns")
        available.difference_update(columns)
        groups[name] = columns
    if available:
        raise ValueError("nomogram terms must cover all non-intercept design columns")
    return groups


def _ticks(distribution, name, requested, n_ticks, adjust_to):
    if requested is None and name not in distribution.values and distribution.limits[name].levels is None:
        requested = np.linspace(*distribution.limits[name].overall, n_ticks)
    return prediction_grid(distribution, {name: requested}, adjust_to=adjust_to)[name]


def _effects(distribution, terms, design, coefficients, columns, intercept, n_ticks, adjust_to):
    values = []
    ranges = []
    for name, term in terms.items():
        if not isinstance(term.by, Mapping) or name in term.by or not set(term.by) <= distribution.limits.keys():
            raise ValueError("interacting predictors must be valid variables other than their axis predictor")
        base = _ticks(distribution, name, term.ticks, n_ticks, adjust_to)
        levels = []
        for by, supplied in term.by.items():
            if supplied is None and distribution.limits[by].levels is None:
                raise ValueError(f"give interaction levels for numeric predictor {by!r}")
            levels.append(_ticks(distribution, by, supplied, n_ticks, adjust_to))
        if prod(map(len, levels)) * len(base) > 100_000:
            raise ValueError("nomogram term exceeds 100,000 settings")
        low, high = np.inf, -np.inf
        for profile in product(*levels):
            conditioning = dict(zip(term.by, profile, strict=True))
            grid = prediction_grid(distribution, {name: base, **conditioning}, adjust_to=adjust_to)
            matrix = np.asarray(design(grid))
            if (matrix.shape != (len(base), len(coefficients)) or matrix.dtype.kind not in "biuf"
                    or not np.isfinite(matrix).all()):
                raise ValueError("design must return one finite column per coefficient on every axis")
            if intercept is not None and not np.all(matrix[:, intercept] == 1):
                raise ValueError("the nominated intercept column must contain ones")
            indices = columns[name]
            effect = np.einsum("ij,j->i", matrix[:, indices], coefficients[indices])
            low, high = min(low, effect.min()), max(high, effect.max())
            values.append((name, conditioning, base.copy(), effect))
        ranges.append((name, low, high))
    return values, ranges


def nomogram(
    distribution: DataDistribution,
    terms: Mapping[str, NomogramTerm],
    design: Callable[[Mapping[str, NDArray]], ArrayLike],
    coefficients: ArrayLike,
    *,
    intercept_column: int | None = 0,
    linear_offset: float = 0,
    adjust_to: Mapping[str, float | str] | None = None,
    max_points: float = 100,
    n_ticks: int = 10,
    linear_ticks: ArrayLike | None = None,
    outcomes: Mapping[str, tuple[Callable, ArrayLike | None]] | None = None,
) -> Nomogram:
    """Construct nomogram scoring axes from a prefit design and its term columns.

    Terms partition non-intercept columns. For an interaction, place the
    relevant main and interaction columns in one term and specify the other
    interacting values in that term's by mapping. Separate profiles share
    the same minimum effect and point scaling for their predictor.
    """
    beta = _numeric_vector(coefficients, "coefficients")
    if not isinstance(distribution, DataDistribution):
        raise ValueError("distribution must be a DataDistribution")
    if not isinstance(terms, Mapping) or not terms or not set(terms) <= distribution.limits.keys():
        raise ValueError("terms must map nominal or numeric variables to their scoring terms")
    if not callable(design):
        raise ValueError("design must be callable")
    if not isinstance(max_points, Real) or not np.isfinite(max_points) or max_points <= 0:
        raise ValueError("max_points must be finite and positive")
    if not isinstance(linear_offset, Real) or not np.isfinite(linear_offset):
        raise ValueError("linear_offset must be finite")
    if not isinstance(n_ticks, Integral) or isinstance(n_ticks, bool) or n_ticks < 2:
        raise ValueError("n_ticks must be an integer of at least two")
    if outcomes is not None and not isinstance(outcomes, Mapping):
        raise ValueError("outcomes must map names to (transform, linear predictor ticks)")
    columns = _term_columns(terms, beta, intercept_column)
    values, ranges = _effects(distribution, terms, design, beta, columns, intercept_column, n_ticks, adjust_to)
    low = {name: minimum for name, minimum, _ in ranges}
    spans = np.array([maximum - minimum for _, minimum, maximum in ranges])
    if not np.isfinite(spans).all() or spans.max() <= 0:
        raise ValueError("at least one term must have a finite, positive effect range")
    scale = max_points / spans.max()
    zero = linear_offset + sum(low.values()) + (0 if intercept_column is None else beta[intercept_column])
    predictors = tuple(
        PointsAxis(name, by, ticks, effect, (effect - low[name]) * scale)
        for name, by, ticks, effect in values
    )
    if linear_ticks is None:
        linear_ticks = np.linspace(zero, zero + spans.sum(), n_ticks)
    linear_ticks = _numeric_vector(linear_ticks, "linear_ticks")
    axes = {}
    for name, (transform, supplied) in (outcomes or {}).items():
        if not callable(transform):
            raise ValueError("each outcome axis must provide a transform")
        ticks = linear_ticks if supplied is None else _numeric_vector(supplied, "outcome linear ticks")
        labels = _numeric_vector(transform(ticks), "outcome axis labels")
        if labels.shape != ticks.shape:
            raise ValueError("outcome transform must return one value per tick")
        axes[name] = OutcomeAxis(labels, ticks.copy(), (ticks - zero) * scale)
    linear_axis = OutcomeAxis(linear_ticks.copy(), linear_ticks, (linear_ticks - zero) * scale)
    return Nomogram(predictors, linear_axis, axes, float(scale), float(zero), float(scale * spans.sum()))
