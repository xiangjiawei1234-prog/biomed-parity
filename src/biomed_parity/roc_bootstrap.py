"""Empirical partial ROC areas and stratified bootstrap inference."""

from dataclasses import dataclass
from math import copysign, inf, nan
from typing import Literal
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .roc import Alternative, _binary_inputs, _p_value, _validate_alternative

RocDirection = Literal["<", ">", "auto"]
Focus = Literal["specificity", "sensitivity"]


@dataclass(frozen=True)
class BootstrapAucResult:
    auc: float
    median: float
    confidence_interval: tuple[float, float]
    n_resamples: int


@dataclass(frozen=True)
class BootstrapPairedResult:
    auc_1: float
    auc_2: float
    difference: float
    standard_error: float
    statistic: float
    p_value: float
    n_resamples: int


@dataclass(frozen=True)
class _EmpiricalROC:
    cases: NDArray[np.bool_]
    rank: NDArray[np.intp]
    size: int

    @classmethod
    def create(cls, labels, scores, direction):
        labels, scores = _binary_inputs(labels, scores)
        cases = labels == 1
        if not cases.any() or cases.all():
            raise ValueError("ROC requires at least one case and one control")
        if direction not in ("<", ">", "auto"):
            raise ValueError("direction must be '<', '>', or 'auto'")
        if direction == "auto":
            direction = "<" if np.median(scores[~cases]) <= np.median(scores[cases]) else ">"
        unique_scores, rank = np.unique(scores, return_inverse=True)
        if direction == "<":
            rank = unique_scores.size - 1 - rank
        return cls(cases, rank, unique_scores.size)

    def counts(self, sample=None):
        cases, rank = (self.cases, self.rank) if sample is None else (
            self.cases[sample], self.rank[sample]
        )
        return (
            np.bincount(rank[cases], minlength=self.size),
            np.bincount(rank[~cases], minlength=self.size),
        )


@dataclass(frozen=True)
class _AreaSpec:
    bounds: tuple[float, float] | None
    focus: Focus
    standardize: bool
    scale: float

    @classmethod
    def create(cls, partial_auc, focus, standardize, percent):
        if focus not in ("specificity", "sensitivity"):
            raise ValueError("focus must be 'specificity' or 'sensitivity'")
        scale = 100.0 if percent else 1.0
        bounds = None
        if partial_auc is not None:
            values = np.asarray(partial_auc, dtype=float)
            if (
                values.shape != (2,) or not np.isfinite(values).all()
                or (values < 0).any() or (values > scale).any()
            ):
                raise ValueError(f"partial_auc must contain two bounds in [0, {scale:g}]")
            low, high = sorted(values / scale)
            bounds = float(low), float(high)
        return cls(bounds, focus, standardize, scale)

    def auc(self, roc: _EmpiricalROC, sample=None, allow_below_chance=False) -> float:
        cases, controls = roc.counts(sample)
        cases_total, controls_total = int(cases.sum()), int(controls.sum())
        if self.bounds is None:
            twice_wins = int(np.dot(controls, 2 * np.cumsum(cases) - cases))
            return self.scale * twice_wins / (2 * cases_total * controls_total)

        low, high = self.bounds
        if low == high:
            return nan if self.standardize else 0.0
        sensitivity = np.concatenate(([0], np.cumsum(cases))) / cases_total
        specificity = (controls_total - np.concatenate(([0], np.cumsum(controls)))) / controls_total
        if self.focus == "specificity":
            axis, height = specificity[::-1], sensitivity[::-1]
        else:
            axis, height = sensitivity, specificity
        area = _restricted_trapezoids(axis, height, low, high)
        if self.standardize:
            width = high - low
            midpoint = (low + high) / 2
            gained = area - width * (1 - midpoint)
            # Guard roundoff at the chance diagonal without masking genuine deficits.
            if abs(gained) <= 8 * np.finfo(float).eps * width:
                gained = 0.0
            if gained < 0 and not allow_below_chance:
                warn(
                    "partial AUC correction is undefined below the chance diagonal",
                    RuntimeWarning, stacklevel=3,
                )
                return nan
            maximal_gain = width * midpoint
            area = 0.5 * (1 + gained / maximal_gain) if maximal_gain else nan
        return self.scale * area


def _restricted_trapezoids(axis, height, low, high):
    mask = (axis[1:] > axis[:-1]) & (axis[1:] > low) & (axis[:-1] < high)
    left, right = axis[:-1][mask], axis[1:][mask]
    first, last = height[:-1][mask], height[1:][mask]
    start, end = np.maximum(low, left), np.minimum(high, right)
    midpoint = (start + end) / 2
    interpolated_height = first + (last - first) * (midpoint - left) / (right - left)
    return float(np.sum((end - start) * interpolated_height))


def _bootstrap_draws(cases, n_resamples, random_state, bootstrap_indices):
    cases_index, controls_index = np.flatnonzero(cases), np.flatnonzero(~cases)
    if bootstrap_indices is not None:
        indices = np.asarray(bootstrap_indices)
        if indices.ndim != 2 or indices.shape[0] < 2 or indices.shape[1] != cases.size:
            raise ValueError("bootstrap_indices must have at least two rows and one column per subject")
        if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= cases.size).any():
            raise ValueError("bootstrap_indices must contain in-range zero-based integer indices")
        if np.any(cases[indices].sum(axis=1) != cases_index.size):
            raise ValueError("each bootstrap draw must preserve the original class counts")
        return indices, indices.shape[0]

    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    rng = np.random.default_rng(random_state)

    def samples():
        for _ in range(n_resamples):
            yield np.concatenate((
                rng.choice(controls_index, size=controls_index.size, replace=True),
                rng.choice(cases_index, size=cases_index.size, replace=True),
            ))

    return samples(), n_resamples


def _finite_bootstrap(values):
    kept = values[np.isfinite(values)]
    if kept.size != values.size:
        warn("nonfinite bootstrap draws were omitted", RuntimeWarning, stacklevel=3)
    return kept


def roc_auc(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    direction: RocDirection = "<",
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
) -> float:
    """Compute empirical binary ROC AUC or a partial area on either axis."""
    roc = _EmpiricalROC.create(labels, scores, direction)
    spec = _AreaSpec.create(partial_auc, focus, standardize, percent)
    return spec.auc(roc, allow_below_chance=allow_below_chance)


def bootstrap_auc_ci(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    direction: RocDirection = "<",
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices: ArrayLike | None = None,
) -> BootstrapAucResult:
    """Return a stratified percentile interval with the observed AUC and bootstrap median."""
    if not 0 <= confidence_level <= 1:
        raise ValueError("confidence_level must lie in [0, 1]")
    roc = _EmpiricalROC.create(labels, scores, direction)
    spec = _AreaSpec.create(partial_auc, focus, standardize, percent)
    draws, n_draws = _bootstrap_draws(roc.cases, n_resamples, random_state, bootstrap_indices)
    observed = spec.auc(roc, allow_below_chance=allow_below_chance)
    values = _finite_bootstrap(np.fromiter(
        (spec.auc(roc, sample, allow_below_chance=True) for sample in draws), float, n_draws
    ))
    tail = (1 - confidence_level) / 2
    lower, median, upper = (
        np.quantile(values, [tail, 0.5, 1 - tail]) if values.size else (nan, nan, nan)
    )
    return BootstrapAucResult(observed, float(median), (float(lower), float(upper)), values.size)


def bootstrap_paired_test(
    labels: ArrayLike,
    scores_1: ArrayLike,
    scores_2: ArrayLike,
    *,
    direction: RocDirection = "<",
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
    alternative: Alternative = "two.sided",
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices: ArrayLike | None = None,
) -> BootstrapPairedResult:
    """Compare two aligned ROC score vectors by pROC's paired bootstrap test."""
    _validate_alternative(alternative)
    first = _EmpiricalROC.create(labels, scores_1, direction)
    second = _EmpiricalROC.create(labels, scores_2, direction)
    spec = _AreaSpec.create(partial_auc, focus, standardize, percent)
    draws, n_draws = _bootstrap_draws(first.cases, n_resamples, random_state, bootstrap_indices)
    auc_1 = spec.auc(first, allow_below_chance=allow_below_chance)
    auc_2 = spec.auc(second, allow_below_chance=allow_below_chance)
    difference = auc_1 - auc_2
    differences = _finite_bootstrap(np.fromiter((
        spec.auc(first, sample, allow_below_chance=True)
        - spec.auc(second, sample, allow_below_chance=True)
        for sample in draws
    ), float, n_draws))
    standard_error, statistic = nan, nan
    if differences.size >= 2:
        standard_error = float(differences.std(ddof=1)) if np.ptp(differences) else 0.0
        if standard_error:
            statistic = difference / standard_error
        elif np.isfinite(difference) and difference != 0:
            statistic = copysign(inf, difference)
        elif difference == 0 and (differences == 0).all():
            statistic = 0.0
    return BootstrapPairedResult(
        auc_1, auc_2, difference, standard_error, statistic,
        _p_value(statistic, norm, alternative), differences.size,
    )
