"""Predictor summaries, prediction grids, and pointwise Wald inference."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from numbers import Integral, Real
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri, stdtrit


@dataclass(frozen=True)
class PredictorLimits:
    """Adjustment value and the effect, display, and observed ranges."""

    effect: tuple[float, float] | None
    adjust_to: float | str
    display: tuple[float | str, float | str]
    overall: tuple[float | str, float | str]
    levels: tuple[float | str, ...] | None = None


@dataclass(frozen=True)
class DataDistribution:
    """Named predictor limits and discrete values in input column order."""

    limits: dict[str, PredictorLimits]
    values: dict[str, tuple[float | str, ...]]


@dataclass(frozen=True)
class WaldPrediction:
    """Point estimates and confidence limits, with standard errors on the link scale."""

    linear_predictor: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    estimate: NDArray[np.float64]
    lower: NDArray[np.float64] | None
    upper: NDArray[np.float64] | None


@dataclass(frozen=True)
class GridPrediction(WaldPrediction):
    """Pointwise predictions and their input settings."""

    grid: dict[str, NDArray]


def _finite_real(value):
    return isinstance(value, (Real, np.bool_)) and np.isfinite(value)


def _missing(value):
    return value is None or (isinstance(value, Real) and np.isnan(value))


def _finite_array(values, name, ndim):
    values = np.asarray(values)
    if values.ndim != ndim or not values.size or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite numeric array with {ndim} dimensions")
    return values.astype(float)


def _probability_pair(values, name):
    values = _finite_array(values, name, 1)
    if len(values) != 2 or not 0 <= values[0] <= values[1] <= 1:
        raise ValueError(f"{name} must contain two ordered probabilities in [0, 1]")
    return values


def _validate_names(mapping, columns, name):
    if not isinstance(mapping, Mapping) or any(key not in columns for key in mapping):
        raise ValueError(f"{name} must map known predictor names to values")


def _factor_summary(observed, specified, name, adjustment):
    if specified is None:
        if not observed or not all(isinstance(value, str) for value in observed):
            raise ValueError(f"factor_levels must be supplied for predictor {name!r}")
        levels = tuple(sorted(set(observed)))
    else:
        values = np.asarray(specified, dtype=object)
        if values.ndim != 1 or not values.size:
            raise ValueError(f"factor_levels for {name!r} must be a nonempty vector")
        levels = tuple(values.tolist())
    if not (
        (all(isinstance(level, str) for level in levels) or all(_finite_real(level) for level in levels))
        and len(set(levels)) == len(levels)
    ):
        raise ValueError(f"factor_levels for {name!r} must be distinct strings or finite numbers")
    counts = dict.fromkeys(levels, 0)
    for value in observed:
        if value not in counts:
            raise ValueError(f"predictor {name!r} contains a value outside its declared factor_levels")
        counts[value] += 1
    adjust_to = levels[0] if adjustment == "first" else max(counts, key=counts.get)
    limits = (levels[0], levels[-1])
    return PredictorLimits(None, adjust_to, limits, limits, levels)


def _quantile_range(observed, quantiles, overall):
    low, high = np.quantile(observed, quantiles)
    return overall if low == high else (float(low), float(high))


def datadist(
    data: Mapping[str, ArrayLike],
    *,
    factor_levels: Mapping[str, ArrayLike] | None = None,
    q_display: ArrayLike | None = None,
    q_effect: ArrayLike = (0.25, 0.75),
    categorical_adjustment: str = "mode",
    n_unique: int = 10,
) -> DataDistribution:
    """Summarize predictor vectors using rms datadist conventions.

    Strings infer nominal factor levels in sorted order. Use factor_levels
    to give a specific order, include unused levels, or designate numeric
    labels as categorical. None and NaN are treated as missing.
    """
    if not isinstance(data, Mapping) or not data or any(not isinstance(name, str) or not name for name in data):
        raise ValueError("data must map nonempty predictor names to observed vectors")
    fixed_levels = {} if factor_levels is None else factor_levels
    _validate_names(fixed_levels, data, "factor_levels")
    if categorical_adjustment not in ("mode", "first"):
        raise ValueError("categorical_adjustment must be 'mode' or 'first'")
    if not isinstance(n_unique, Integral) or isinstance(n_unique, bool) or n_unique < 0:
        raise ValueError("n_unique must be a nonnegative integer")
    display_probs = None if q_display is None else _probability_pair(q_display, "q_display")
    effect_probs = _probability_pair(q_effect, "q_effect")
    limits, values = {}, {}
    expected_count = None
    for name, column in data.items():
        column = np.asarray(column, dtype=object)
        if column.ndim != 1 or not column.size or (expected_count is not None and len(column) != expected_count):
            raise ValueError("data columns must be nonempty one-dimensional vectors of the same length")
        expected_count = len(column)
        observed = [value for value in column if not _missing(value)]
        if name in fixed_levels or (observed and all(isinstance(value, str) for value in observed)):
            summary = _factor_summary(observed, fixed_levels.get(name), name, categorical_adjustment)
            limits[name] = summary
            if len(summary.levels) < len(column):
                values[name] = summary.levels
            continue
        if not all(_finite_real(value) for value in observed):
            raise ValueError(f"predictor {name!r} must contain finite numbers or categorical labels")
        if len(observed) < 2:
            raise ValueError(f"fewer than two nonmissing observations for {name!r}")
        observed = np.asarray(observed, dtype=float)
        unique = np.unique(observed)
        overall = (float(unique[0]), float(unique[-1]))
        if len(unique) < 4:
            effect = display = overall
            adjust = float(unique[(len(unique) - 1) // 2])
            if len(unique) == 1:
                warn(f"predictor {name!r} is constant", stacklevel=2)
        else:
            if display_probs is None:
                tail = 10 / max(len(observed), 200)
                display_probs = (tail, 1 - tail)
            display = _quantile_range(observed, display_probs, overall)
            effect = _quantile_range(observed, effect_probs, overall)
            adjust = float(np.median(observed))
        limits[name] = PredictorLimits(effect, adjust, display, overall)
        if len(unique) <= n_unique:
            values[name] = tuple(unique.tolist())
    return DataDistribution(limits, values)


def _predictor_values(values, name, limits):
    array = np.atleast_1d(np.asarray(values, dtype=object))
    if array.ndim != 1 or not array.size:
        raise ValueError(f"values for {name!r} must be a nonempty vector")
    if limits.levels is None:
        if not all(_finite_real(value) for value in array):
            raise ValueError(f"values for {name!r} must be finite numbers")
        return array.astype(float)
    if any(value not in limits.levels for value in array):
        raise ValueError(f"values for {name!r} must be declared factor levels")
    return np.asarray(array.tolist())


def _adjustment_values(distribution, adjust_to):
    if not isinstance(distribution, DataDistribution):
        raise TypeError("distribution must be a DataDistribution")
    overrides = {} if adjust_to is None else adjust_to
    _validate_names(overrides, distribution.limits, "adjust_to")
    result = {}
    for name, limits in distribution.limits.items():
        value = _predictor_values(overrides.get(name, limits.adjust_to), name, limits)
        if len(value) != 1:
            raise ValueError(f"adjust_to for {name!r} must specify one value")
        result[name] = value
    return result


def _expand_grid(distribution, vary, points, settings):
    _validate_names(vary, distribution.limits, "vary")
    if not isinstance(points, Integral) or isinstance(points, bool) or points < 1:
        raise ValueError("points must be a positive integer")
    names = tuple(vary) if len(vary) == len(settings) else tuple(settings)
    settings = settings.copy()
    for name, values in vary.items():
        limits = distribution.limits[name]
        if values is None:
            values = limits.levels if limits.levels is not None else distribution.values.get(name)
            if values is None:
                if points > 100_000:
                    raise ValueError("prediction grid exceeds 100,000 combinations")
                values = np.linspace(*limits.display, points)
        settings[name] = _predictor_values(values, name, limits)
    combinations = prod(len(settings[name]) for name in names)
    if combinations > 100_000:
        raise ValueError("prediction grid exceeds 100,000 combinations")
    grid = {}
    stride = 1
    for name in names:
        values = settings[name]
        grid[name] = np.tile(np.repeat(values, stride), combinations // (stride * len(values)))
        stride *= len(values)
    return grid


def prediction_grid(
    distribution: DataDistribution,
    vary: Mapping[str, ArrayLike | None],
    *,
    points: int = 200,
    adjust_to: Mapping[str, float | str] | None = None,
) -> dict[str, NDArray]:
    """Build a rectangular settings grid with the first predictor varying fastest.

    Map predictors to explicit values or None for their default grid.
    Unspecified predictors use their adjustment values. An empty mapping
    produces the single adjustment row.
    """
    return _expand_grid(distribution, vary, points, _adjustment_values(distribution, adjust_to))


def wald_predict(
    design: ArrayLike,
    coefficients: ArrayLike,
    covariance: ArrayLike,
    *,
    reference: ArrayLike | None = None,
    center: ArrayLike | None = None,
    confidence_level: float = 0.95,
    residual_df: float | None = None,
    transform: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
) -> WaldPrediction:
    """Evaluate a prefit numeric design and its pointwise mean confidence limits.

    A reference row gives differences from a fixed predictor setting. A
    center row provides the fitted centering design, as for a Cox model.
    Supply residual degrees of freedom for t limits. A monotone transform
    maps estimates and confidence endpoints after link-scale inference.
    """
    design = _finite_array(design, "design", 2)
    beta = _finite_array(coefficients, "coefficients", 1)
    covariance = _finite_array(covariance, "covariance", 2)
    if design.shape[1] != len(beta) or covariance.shape != (len(beta), len(beta)):
        raise ValueError("design columns and covariance must match coefficients")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
        raise ValueError("covariance must be symmetric")
    if reference is not None and center is not None:
        raise ValueError("supply either reference or center")
    if reference is not None or center is not None:
        label = "reference" if reference is not None else "center"
        baseline = _finite_array(reference if reference is not None else center, label, 1)
        if len(baseline) != len(beta):
            raise ValueError(f"{label} must have one entry per coefficient")
        design -= baseline
    linear = design @ beta
    variance = np.einsum("ij,ij->i", design @ covariance, design)
    if np.any(variance < 0):
        magnitude = np.einsum("ij,ij->i", np.abs(design) @ np.abs(covariance), np.abs(design))
        if np.any(variance < -16 * np.finfo(float).eps * magnitude):
            raise ValueError("covariance implies a negative prediction variance")
    error = np.sqrt(np.maximum(variance, 0))
    if not _finite_real(confidence_level) or not 0 <= confidence_level < 1:
        raise ValueError("confidence_level must be in [0, 1)")
    if residual_df is not None and (not _finite_real(residual_df) or residual_df <= 0):
        raise ValueError("residual_df must be positive and finite")

    def output(values):
        result = values if transform is None else np.asarray(transform(values.copy()), dtype=float)
        if result.shape != values.shape or np.isnan(result).any():
            raise ValueError("transform must return one numeric result per prediction")
        return result

    estimate = output(linear)
    if confidence_level == 0:
        return WaldPrediction(linear, error, estimate, None, None)
    tail = (1 - confidence_level) / 2
    critical = -ndtri(tail) if residual_df is None else -stdtrit(residual_df, tail)
    lower, upper = output(linear - critical * error), output(linear + critical * error)
    return WaldPrediction(linear, error, estimate, np.minimum(lower, upper), np.maximum(lower, upper))


def predict(
    distribution: DataDistribution,
    vary: Mapping[str, ArrayLike | None],
    design: Callable[[Mapping[str, NDArray]], ArrayLike],
    coefficients: ArrayLike,
    covariance: ArrayLike,
    *,
    points: int = 200,
    adjust_to: Mapping[str, float | str] | None = None,
    ref_zero: bool = False,
    center: ArrayLike | None = None,
    confidence_level: float = 0.95,
    residual_df: float | None = None,
    transform: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
) -> GridPrediction:
    """Apply the fitted model's design function to an rms-style prediction grid."""
    if not isinstance(ref_zero, (bool, np.bool_)):
        raise ValueError("ref_zero must be boolean")
    adjustment = _adjustment_values(distribution, adjust_to)
    grid = _expand_grid(distribution, vary, points, adjustment)
    matrix = _finite_array(design(grid), "design result", 2)
    if len(matrix) != len(next(iter(grid.values()))):
        raise ValueError("design must return one row per prediction setting")
    reference = None
    if ref_zero:
        baseline = _finite_array(design(adjustment), "reference design", 2)
        if len(baseline) != 1:
            raise ValueError("design must return one row for the adjustment setting")
        reference = baseline[0]
    result = wald_predict(
        matrix, coefficients, covariance, reference=reference, center=None if ref_zero else center,
        confidence_level=confidence_level, residual_df=residual_df, transform=transform,
    )
    return GridPrediction(
        result.linear_predictor, result.standard_error, result.estimate, result.lower, result.upper, grid
    )
