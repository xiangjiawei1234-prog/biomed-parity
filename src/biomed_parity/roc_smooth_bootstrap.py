"""Bootstrap intervals and tests for smoothed binary ROC curves."""

from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import Literal
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .roc import Alternative, _binary_inputs, _p_value, _validate_alternative
from .roc_bootstrap import Focus, RocDirection, _paired_auc_statistic
from .roc_parametric import ParametricDistribution, smooth_parametric_roc
from .roc_smoothing import DensityKernel, SmoothingMethod, smooth_roc
from .roc_thresholds import _draws

BootstrapSmoothingMethod = SmoothingMethod | Literal["fitdistr"]


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


@dataclass(frozen=True)
class SmoothPairedAucResult:
    """Observed areas, paired bootstrap test, and draw counts."""

    auc_1: float
    auc_2: float
    difference: float
    standard_error: float
    statistic: float
    p_value: float
    n_resamples: int
    omitted_resamples: int


@dataclass(frozen=True)
class SmoothUnpairedAucResult:
    """Observed areas, independent-sample bootstrap test, and draw counts."""

    auc_1: float
    auc_2: float
    difference: float
    standard_error: float
    statistic: float
    p_value: float
    n_resamples: int
    omitted_resamples: int


def _probabilities(confidence_level):
    if not isinstance(confidence_level, Real) or not 0 <= confidence_level <= 1:
        raise ValueError("confidence_level must lie in [0, 1]")
    tail = (1 - confidence_level) / 2
    return (tail, 0.5, 1 - tail)


def _fitter(method, bw, kernel, adjust, cut, control_distribution, case_distribution):
    if method == "fitdistr":
        return partial(smooth_parametric_roc, control_distribution=control_distribution,
            case_distribution=case_distribution)
    if method not in ("binormal", "density", "logcondens", "logcondens.smooth"):
        raise ValueError("method must be 'binormal', 'density', 'fitdistr', 'logcondens', or 'logcondens.smooth'")
    return partial(smooth_roc, method=method, bw=bw, kernel=kernel, adjust=adjust, cut=cut)


def _refit_samples(labels, estimate, stratified, n_resamples, random_state, indices):
    rng = np.random.default_rng(random_state) if indices is None else None
    draws, count = _draws(labels == 1, n_resamples, rng, indices, stratified)
    return _refit_draws(draws, count, estimate)


def _refit_draws(draws, count, estimate):
    values = []
    for selection in draws:
        try:
            value = estimate(selection)
        except ValueError:
            # The full sample has already validated settings. Resampling can
            # remove a class or leave too little variation to fit a curve.
            continue
        if np.isfinite(value).all():
            values.append(value)
    omitted = count - len(values)
    if omitted:
        warn(f"{omitted} bootstrap draws with no usable smoothed estimates were omitted",
            RuntimeWarning, stacklevel=4)
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
    method: BootstrapSmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    control_distribution: ParametricDistribution = "normal",
    case_distribution: ParametricDistribution = "normal",
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
    With method="fitdistr", independently refit the chosen class distributions.
    """
    probabilities = _probabilities(confidence_level)
    labels, scores = _binary_inputs(labels, scores)
    fit = _fitter(method, bw, kernel, adjust, cut, control_distribution, case_distribution)
    settings = dict(direction=direction, n=n, partial_auc=partial_auc, focus=focus, standardize=standardize,
        allow_below_chance=allow_below_chance, percent=percent)
    observed = fit(labels, scores, **settings)
    scale = 100.0 if percent else 1.0
    if percent and partial_auc is not None:
        partial_auc = np.asarray(partial_auc, dtype=float) / scale
    settings.update(direction=observed.direction, allow_below_chance=True,
        percent=False, partial_auc=partial_auc)

    def estimate(selection):
        return fit(labels[selection], scores[selection], **settings).auc

    values, count, omitted = _refit_samples(labels, estimate, stratified, n_resamples,
        random_state, bootstrap_indices)
    lower, median, upper = np.quantile(values, probabilities) * scale if values else (np.nan,) * 3
    return SmoothAucCIResult(observed.auc, float(median), (float(lower), float(upper)), count, omitted)


def bootstrap_smooth_sensitivity_ci(
    labels: ArrayLike,
    scores: ArrayLike,
    specificity: ArrayLike,
    *,
    method: BootstrapSmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    control_distribution: ParametricDistribution = "normal",
    case_distribution: ParametricDistribution = "normal",
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
    With method="fitdistr", independently refit the chosen class distributions.
    """
    probabilities = _probabilities(confidence_level)
    labels, scores = _binary_inputs(labels, scores)
    scale = 100.0 if percent else 1.0
    targets = np.atleast_1d(np.asarray(specificity))
    if (targets.ndim != 1 or not targets.size or targets.dtype.kind not in "iuf"
        or not np.isfinite(targets).all() or np.any((targets < 0) | (targets > scale))):
        raise ValueError(f"specificity must contain finite values in [0, {scale:g}]")
    targets = targets.astype(float)
    fit = _fitter(method, bw, kernel, adjust, cut, control_distribution, case_distribution)
    settings = dict(direction=direction, n=n, percent=percent)
    observed = fit(labels, scores, **settings)
    settings["direction"] = observed.direction
    sensitivity = _sensitivity_at(observed, targets)

    def estimate(selection):
        curve = fit(labels[selection], scores[selection], **settings)
        return _sensitivity_at(curve, targets)

    values, count, omitted = _refit_samples(labels, estimate, stratified, n_resamples,
        random_state, bootstrap_indices)
    intervals = np.quantile(values, probabilities, axis=0).T if values else np.full((targets.size, 3), np.nan)
    return SmoothSensitivityCIResult(targets, sensitivity, intervals, count, omitted)


def bootstrap_smooth_paired_test(
    labels: ArrayLike,
    scores_1: ArrayLike,
    scores_2: ArrayLike,
    *,
    method: BootstrapSmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    control_distribution: ParametricDistribution = "normal",
    case_distribution: ParametricDistribution = "normal",
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
    alternative: Alternative = "two.sided",
    stratified: bool = True,
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices: ArrayLike | None = None,
) -> SmoothPairedAucResult:
    """Compare smoothed AUCs from two score vectors aligned by subject.

    Each draw refits both curves on the same subjects. Automatic directions
    are resolved separately on the original sample and then held fixed.
    The statistic uses AUC 1 minus AUC 2 and the sample SD of the paired draws.
    """
    _validate_alternative(alternative)
    labels, scores_1 = _binary_inputs(labels, scores_1)
    _, scores_2 = _binary_inputs(labels, scores_2)
    fit = _fitter(method, bw, kernel, adjust, cut, control_distribution, case_distribution)
    settings = dict(direction=direction, n=n, partial_auc=partial_auc, focus=focus, standardize=standardize,
        allow_below_chance=allow_below_chance, percent=percent)
    return SmoothPairedAucResult(*_smooth_auc_test(
        (labels, scores_1), (labels, scores_2), fit, settings, alternative,
        stratified, n_resamples, random_state, bootstrap_indices, None, paired=True,
    ))


def _comparison_draws(cases_1, cases_2, stratified, n_resamples, random_state, indices_1, indices_2, *, paired):
    uses_rng = indices_1 is None or (not paired and indices_2 is None)
    rng = np.random.default_rng(random_state) if uses_rng else None
    if paired:
        draws, count = _draws(cases_1, n_resamples, rng, indices_1, stratified)
        return ((selection, selection) for selection in draws), count

    if indices_1 is None and indices_2 is not None:
        draws_2, count = _draws(cases_2, n_resamples, rng, indices_2, stratified)
        draws_1, _ = _draws(cases_1, count, rng, None, stratified)
    else:
        draws_1, count = _draws(cases_1, n_resamples, rng, indices_1, stratified)
        draws_2, count_2 = _draws(cases_2, count, rng, indices_2, stratified)
        if count != count_2:
            raise ValueError("bootstrap index matrices must have the same number of rows")
    return zip(draws_1, draws_2, strict=True), count


def _smooth_auc_test(first_data, second_data, fit, settings, alternative,
    stratified, n_resamples, random_state, indices_1, indices_2, *, paired):
    labels_1, scores_1 = first_data
    labels_2, scores_2 = second_data
    first = fit(labels_1, scores_1, **settings)
    second = fit(labels_2, scores_2, **settings)
    difference = first.auc - second.auc
    draw_settings = dict(settings, allow_below_chance=True)
    draw_settings.pop("direction")

    def estimate(selection):
        selection_1, selection_2 = selection
        auc_1 = fit(labels_1[selection_1], scores_1[selection_1], direction=first.direction, **draw_settings).auc
        auc_2 = fit(labels_2[selection_2], scores_2[selection_2], direction=second.direction, **draw_settings).auc
        return auc_1 - auc_2

    draws, count = _comparison_draws(labels_1 == 1, labels_2 == 1, stratified, n_resamples,
        random_state, indices_1, indices_2, paired=paired)
    values, count, omitted = _refit_draws(draws, count, estimate)
    standard_error, statistic = _paired_auc_statistic(difference, np.asarray(values))
    return (first.auc, second.auc, difference, standard_error, statistic,
        _p_value(statistic, norm, alternative), count, omitted)


def bootstrap_smooth_unpaired_test(
    labels_1: ArrayLike,
    scores_1: ArrayLike,
    labels_2: ArrayLike,
    scores_2: ArrayLike,
    *,
    method: BootstrapSmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    control_distribution: ParametricDistribution = "normal",
    case_distribution: ParametricDistribution = "normal",
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
    alternative: Alternative = "two.sided",
    stratified: bool = True,
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices_1: ArrayLike | None = None,
    bootstrap_indices_2: ArrayLike | None = None,
) -> SmoothUnpairedAucResult:
    """Compare smoothed AUCs from two independent samples.

    Each replicate draws from each sample independently and refits both
    curves, holding their original directions fixed. The standard error
    is the sample SD of the resulting AUC differences. A supplied index
    matrix sets the replicate count for both samples.
    """
    _validate_alternative(alternative)
    first = _binary_inputs(labels_1, scores_1)
    second = _binary_inputs(labels_2, scores_2)
    fit = _fitter(method, bw, kernel, adjust, cut, control_distribution, case_distribution)
    settings = dict(direction=direction, n=n, partial_auc=partial_auc, focus=focus, standardize=standardize,
        allow_below_chance=allow_below_chance, percent=percent)
    return SmoothUnpairedAucResult(*_smooth_auc_test(
        first, second, fit, settings, alternative, stratified, n_resamples, random_state,
        bootstrap_indices_1, bootstrap_indices_2, paired=False,
    ))
