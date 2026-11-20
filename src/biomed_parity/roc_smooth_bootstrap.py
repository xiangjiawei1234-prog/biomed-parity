"""Percentile bootstrap intervals for one smoothed binary ROC curve."""

from dataclasses import dataclass
from numbers import Real
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .roc import _binary_inputs
from .roc_bootstrap import Focus, RocDirection
from .roc_smoothing import DensityKernel, SmoothingMethod, smooth_roc
from .roc_thresholds import _draws


@dataclass(frozen=True)
class SmoothAucCIResult:
    """Observed area, bootstrap percentiles, and draw counts."""

    auc: float
    median: float
    confidence_interval: tuple[float, float]
    n_resamples: int
    omitted_resamples: int


@dataclass(frozen=True)
class SmoothSensitivityCIResult:
    """Observed sensitivity and lower, median, and upper values per specificity."""

    specificity: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    intervals: NDArray[np.float64]
    n_resamples: int
    omitted_resamples: int


def _probabilities(confidence_level):
    if not isinstance(confidence_level, Real) or not 0 <= confidence_level <= 1:
        raise ValueError("confidence_level must lie in [0, 1]")
    tail = (1 - confidence_level) / 2
    return (tail, 0.5, 1 - tail)


def _refit_samples(labels, scores, settings, estimate, stratified, n_resamples, random_state, indices):
    rng = np.random.default_rng(random_state) if indices is None else None
    draws, count = _draws(labels == 1, n_resamples, rng, indices, stratified)
    values = []
    for selection in draws:
        try:
            refit = smooth_roc(labels[selection], scores[selection], **settings)
        except ValueError:
            # The full sample has already validated settings. Resampling can
            # remove a class or leave too few interior points for a binormal fit.
            continue
        value = estimate(refit)
        if np.isfinite(value).all():
            values.append(value)
    omitted = count - len(values)
    if omitted:
        warn(f"{omitted} bootstrap draws with no usable smoothed estimates were omitted",
            RuntimeWarning, stacklevel=3)
    return values, count, omitted


def _sensitivity_at(curve, targets):
    sp, se = curve.specificity, curve.sensitivity
    upper = np.searchsorted(sp, targets, side="left")
    lower = np.maximum(upper - 1, 0)
    # Exact matches select the first point, including at vertical curve segments.
    fraction = np.divide(targets - sp[upper], sp[lower] - sp[upper],
        where=targets != sp[upper], out=np.zeros_like(targets))
    return se[upper] + fraction * (se[lower] - se[upper])


def bootstrap_smooth_auc_ci(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    method: SmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
    stratified: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices: ArrayLike | None = None,
) -> SmoothAucCIResult:
    """Refit each bootstrap sample and return a percentile interval for smoothed AUC.

    Sampling is stratified by default. Fixed zero-based index rows override
    n_resamples. All draws use the direction resolved on the original sample.
    """
    probabilities = _probabilities(confidence_level)
    labels, scores = _binary_inputs(labels, scores)
    settings = dict(method=method, direction=direction, n=n, bw=bw, kernel=kernel,
        adjust=adjust, cut=cut, partial_auc=partial_auc, focus=focus, standardize=standardize,
        allow_below_chance=allow_below_chance, percent=percent)
    observed = smooth_roc(labels, scores, **settings)
    scale = 100.0 if percent else 1.0
    if percent and partial_auc is not None:
        partial_auc = np.asarray(partial_auc, dtype=float) / scale
    settings.update(direction=observed.direction, allow_below_chance=True,
        percent=False, partial_auc=partial_auc)
    values, count, omitted = _refit_samples(labels, scores, settings,
        lambda result: result.auc, stratified, n_resamples, random_state, bootstrap_indices)
    lower, median, upper = np.quantile(values, probabilities) * scale if values else (np.nan,) * 3
    return SmoothAucCIResult(observed.auc, float(median), (float(lower), float(upper)), count, omitted)


def bootstrap_smooth_sensitivity_ci(
    labels: ArrayLike,
    scores: ArrayLike,
    specificity: ArrayLike,
    *,
    method: SmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    percent: bool = False,
    stratified: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices: ArrayLike | None = None,
) -> SmoothSensitivityCIResult:
    """Return pointwise bootstrap intervals at fixed specificity values.

    Coordinates use the same scale as specificity. Fixed zero-based index
    rows override n_resamples. Each draw refits the original smoothing method.
    """
    probabilities = _probabilities(confidence_level)
    labels, scores = _binary_inputs(labels, scores)
    scale = 100.0 if percent else 1.0
    targets = np.atleast_1d(np.asarray(specificity))
    if (targets.ndim != 1 or not targets.size or targets.dtype.kind not in "iuf"
        or not np.isfinite(targets).all() or np.any((targets < 0) | (targets > scale))):
        raise ValueError(f"specificity must contain finite values in [0, {scale:g}]")
    targets = targets.astype(float)
    settings = dict(method=method, direction=direction, n=n, bw=bw, kernel=kernel,
        adjust=adjust, cut=cut, percent=percent)
    observed = smooth_roc(labels, scores, **settings)
    settings["direction"] = observed.direction
    sensitivity = _sensitivity_at(observed, targets)
    values, count, omitted = _refit_samples(labels, scores, settings,
        lambda result: _sensitivity_at(result, targets), stratified, n_resamples, random_state, bootstrap_indices)
    intervals = np.quantile(values, probabilities, axis=0).T if values else np.full((targets.size, 3), np.nan)
    return SmoothSensitivityCIResult(targets, sensitivity, intervals, count, omitted)
