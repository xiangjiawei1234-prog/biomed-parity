"""Empirical ROC operating points, cutoffs, and percentile intervals."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .roc import _binary_inputs
from .roc_bootstrap import Focus, RocDirection, _AreaSpec, _bootstrap_draws

_COORDINATES = (
    "threshold", "specificity", "sensitivity", "accuracy", "tn", "tp", "fn", "fp",
    "npv", "ppv", "fdr", "fpr", "tpr", "tnr", "fnr", "1-specificity", "1-sensitivity",
    "1-accuracy", "1-npv", "1-ppv", "precision", "recall", "lr_pos", "lr_neg", "youden",
    "closest.topleft",
)
_DEFAULT_RET = ("threshold", "specificity", "sensitivity")
_ASCENDING = {"specificity", "tn", "fn", "tnr", "fnr", "1-sensitivity"}
_DESCENDING = {"sensitivity", "tp", "fp", "tpr", "fpr", "recall", "1-specificity"}
_ALIASES = {"t": "threshold", "topleft": "closest.topleft", "npe": "1-npv", "ppe": "1-ppv"}
BestMethod = Literal["youden", "closest.topleft"]


@dataclass(frozen=True)
class BootstrapCoordsResult:
    """One quantile interval per requested point and metric."""

    fields: tuple[str, ...]
    observed: NDArray[np.float64]
    intervals: NDArray[np.float64]
    counts: NDArray[np.int64]
    n_resamples: int
    omitted_resamples: int


@dataclass(frozen=True)
class _ROCPoints:
    threshold: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    specificity: NDArray[np.float64]
    n_cases: int
    n_controls: int
    scale: float
    direction: str
    oriented_scores: NDArray[np.float64] | None


def _resolve_direction(cases, scores, direction):
    if direction not in ("<", ">", "auto"):
        raise ValueError("direction must be '<', '>', or 'auto'")
    if direction == "auto":
        return "<" if np.median(scores[~cases]) <= np.median(scores[cases]) else ">"
    return direction


def _empirical_points(labels, scores, direction, percent):
    cases = labels == 1
    n_cases, n_controls = int(cases.sum()), int((~cases).sum())
    if not n_cases or not n_controls:
        raise ValueError("ROC requires at least one case and one control")
    direction = _resolve_direction(cases, scores, direction)
    sign = 1 if direction == "<" else -1
    unique, inverse = np.unique(sign * scores.astype(float), return_inverse=True)
    with np.errstate(over="ignore"):
        midpoints = (unique[:-1] + unique[1:]) / 2
    half_midpoints = unique[:-1] / 2 + unique[1:] / 2
    midpoints = np.where(np.abs(midpoints) > 1e100, half_midpoints, midpoints)
    midpoints = np.where(midpoints == unique[:-1], unique[1:], midpoints)
    scale = 100.0 if percent else 1.0
    passed_cases = np.r_[0, np.cumsum(np.bincount(inverse[cases], minlength=len(unique)))]
    passed_controls = np.r_[0, np.cumsum(np.bincount(inverse[~cases], minlength=len(unique)))]
    return _ROCPoints(
        sign * np.r_[-np.inf, midpoints, np.inf],
        (n_cases - passed_cases) / n_cases * scale,
        passed_controls / n_controls * scale,
        n_cases, n_controls, scale, direction, unique,
    )


def _fields(ret, has_threshold=True):
    if isinstance(ret, str):
        if ret == "all":
            return _COORDINATES if has_threshold else _COORDINATES[1:]
        ret = (ret,)
    raw = tuple(ret)
    if raw == ("all",):
        return _fields("all", has_threshold)
    names = tuple(_ALIASES.get(name, name) for name in raw)
    allowed = _COORDINATES if has_threshold else _COORDINATES[1:]
    if not names or any(name not in allowed for name in names):
        raise ValueError("ret contains an unsupported ROC coordinate")
    return names


def _weight_ratio(weights):
    values = np.asarray(weights)
    if values.shape != (2,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("best_weights must contain a finite cost and prevalence")
    cost, prevalence = map(float, values)
    if cost <= 0 or not 0 < prevalence < 1:
        raise ValueError("best_weights requires positive cost and prevalence in (0, 1)")
    return (1 - prevalence) / (cost * prevalence)


def _metric_values(points, threshold, se, sp, ratio):
    scale = points.scale
    tp = se * points.n_cases / scale
    tn = sp * points.n_controls / scale
    fn, fp = points.n_cases - tp, points.n_controls - tn
    accuracy = scale * (tp + tn) / (points.n_cases + points.n_controls)
    with np.errstate(invalid="ignore", divide="ignore"):
        ppv = scale * tp / (tp + fp)
        npv = scale * tn / (tn + fn)
        fdr = scale * fp / (tp + fp)
        fnr = scale * fn / (tp + fn)
        lr_pos = (se / scale) / (1 - sp / scale)
        lr_neg = (1 - se / scale) / (sp / scale)
    return {
        "threshold": threshold, "sensitivity": se, "specificity": sp,
        "accuracy": accuracy, "tn": tn, "tp": tp, "fn": fn, "fp": fp,
        "npv": npv, "ppv": ppv, "tpr": se, "tnr": sp, "fpr": scale - sp, "fnr": fnr, "fdr": fdr,
        "1-specificity": scale - sp, "1-sensitivity": scale - se, "1-accuracy": scale - accuracy,
        "1-npv": scale - npv, "1-ppv": scale - ppv, "precision": ppv, "recall": se,
        "lr_pos": lr_pos, "lr_neg": lr_neg, "youden": se + ratio * sp,
        "closest.topleft": ((scale - se)**2 + ratio * (scale - sp)**2) / scale,
    }


def _numeric_coords(points, requested, input_name, ratio):
    values = np.atleast_1d(np.asarray(requested))
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iuf" or np.isnan(values).any():
        raise ValueError("x must contain numeric ROC coordinate values")
    values = values.astype(float)
    if input_name == "threshold":
        if points.oriented_scores is None:
            raise ValueError("smoothed ROC coordinates have no thresholds")
        sign = 1 if points.direction == "<" else -1
        index = np.searchsorted(points.oriented_scores, sign * values, side="left")
        return values, points.sensitivity[index], points.specificity[index]

    threshold = points.threshold
    se, sp = points.sensitivity, points.specificity
    if input_name == "sensitivity":
        observed = se
    elif input_name == "specificity":
        observed = sp
    elif input_name in _ASCENDING | _DESCENDING:
        observed = _metric_values(points, threshold, se, sp, ratio)[input_name]
        threshold = np.full(len(se), np.nan)
    else:
        raise ValueError("input must name a monotone ROC coordinate")

    # Look at an increasing axis whose first match maximizes the other rate.
    if input_name in _DESCENDING:
        observed, threshold, se, sp = observed[::-1], threshold[::-1], se[::-1], sp[::-1]
    if np.any(values < observed[0]) or np.any(values > observed[-1]):
        raise ValueError(f"input {input_name} is outside [{observed[0]:g}, {observed[-1]:g}]")
    upper = np.searchsorted(observed, values, side="left")
    exact = observed[upper] == values
    lower = np.where(exact, upper, upper - 1)
    width = observed[upper] - observed[lower]
    fraction = np.divide(values - observed[lower], width, out=np.zeros_like(values), where=~exact)
    return (
        np.where(exact, threshold[upper], np.nan),
        se[lower] + fraction * (se[upper] - se[lower]),
        sp[lower] + fraction * (sp[upper] - sp[lower]),
    )


def _local_maxima(specificity, sensitivity):
    count = len(specificity)
    if count == 1:
        return np.ones(1, dtype=bool)
    sp_gain = np.diff(specificity) > 0
    se_drop = np.diff(sensitivity) < 0
    interior = (sp_gain[:-1] & se_drop[1:]) | (sp_gain[:-1] & (sensitivity[1:-1] == 0))
    interior |= (~se_drop[:-1] & (sensitivity[1:-1] > sensitivity[:-2]) & (specificity[1:-1] == 1))
    return np.r_[se_drop[0], interior, sp_gain[-1]]


def _curve_coords(points, x, input_name, fields, best_method, ratio, area):
    if isinstance(x, str):
        if x not in ("all", "best", "local maximas"):
            raise ValueError("x must be 'all', 'best', 'local maximas', or numeric values")
        th, se, sp = points.threshold, points.sensitivity, points.specificity
        if area.bounds is not None:
            low, high = (bound * points.scale for bound in area.bounds)
            axis = se if area.focus == "sensitivity" else sp
            keep = (axis >= low) & (axis <= high)
            th, se, sp, axis = th[keep], se[keep], sp[keep], axis[keep]
            if x == "all":
                endpoints = (high, low) if area.focus == "sensitivity" else (low, high)
                rows = np.column_stack((th, se, sp))
                for position, value in enumerate(endpoints):
                    if not np.any(axis == value):
                        boundary = np.column_stack(_numeric_coords(points, [value], area.focus, ratio))
                        rows = np.concatenate((rows, boundary) if position else (boundary, rows))
                th, se, sp = rows.T
        if x == "local maximas" and len(th):
            distinct = np.r_[True, (np.diff(se) != 0) | (np.diff(sp) != 0)]
            picked = np.flatnonzero(distinct)[_local_maxima(sp[distinct], se[distinct])]
            th, se, sp = th[picked], se[picked], sp[picked]
        if x == "best" and len(th):
            criterion = se + ratio * sp if best_method == "youden" else -(
                (points.scale - se)**2 + ratio * (points.scale - sp)**2
            )
            selected = criterion == criterion.max()
            th, se, sp = th[selected], se[selected], sp[selected]
    else:
        th, se, sp = _numeric_coords(points, x, input_name, ratio)
    metrics = _metric_values(points, th, se, sp, ratio)
    # pROC reuses the raw optimization criterion when only core columns
    # and that criterion are requested. Derived percent-scale fields differ.
    if (
        isinstance(x, str) and x == "best" and best_method == "closest.topleft"
        and set(fields).issubset(set(_DEFAULT_RET) | {"closest.topleft"})
    ):
        metrics["closest.topleft"] *= points.scale
    return np.column_stack(tuple(metrics[name] for name in fields))


def roc_coords(
    labels: ArrayLike,
    scores: ArrayLike,
    x: str | ArrayLike = "all",
    *,
    input: str = "threshold",
    ret: str | tuple[str, ...] = _DEFAULT_RET,
    direction: RocDirection = "<",
    best_method: BestMethod = "youden",
    best_weights: tuple[float, float] = (1, 0.5),
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    percent: bool = False,
) -> NDArray[np.float64]:
    """Return rows of requested ROC coordinates in the order given by ret."""
    labels, scores = _binary_inputs(labels, scores)
    points = _empirical_points(labels, scores, direction, percent)
    fields = _fields(ret)
    ratio = _weight_ratio(best_weights)
    if best_method not in ("youden", "closest.topleft"):
        raise ValueError("best_method must be 'youden' or 'closest.topleft'")
    area = _AreaSpec.create(partial_auc, focus, False, percent)
    return _curve_coords(points, x, _ALIASES.get(input, input), fields, best_method, ratio, area)


def _draws(cases, n_resamples, rng, bootstrap_indices, stratified):
    if stratified:
        return _bootstrap_draws(cases, n_resamples, rng, bootstrap_indices)
    if bootstrap_indices is not None:
        indices = np.asarray(bootstrap_indices)
        if indices.ndim != 2 or indices.shape[0] < 2 or indices.shape[1] != len(cases):
            raise ValueError("bootstrap_indices must have at least two rows and one column per subject")
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(cases)):
            raise ValueError("bootstrap_indices must contain in-range zero-based integer indices")
        return indices, len(indices)
    if not isinstance(n_resamples, Integral) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    draws = (rng.integers(0, len(cases), len(cases)) for _ in range(n_resamples))
    return draws, n_resamples


def _quantile_with_infinities(values, probabilities):
    valid = np.sort(values[~np.isnan(values)])
    if not len(valid):
        return np.full(3, np.nan)
    ranks = probabilities * (len(valid) - 1)
    left, right = valid[np.floor(ranks).astype(int)], valid[np.ceil(ranks).astype(int)]
    fraction = ranks - np.floor(ranks)
    result = left.copy()
    between = (fraction != 0) & (left != right)
    with np.errstate(invalid="ignore"):
        result[between] = left[between] * (1 - fraction[between]) + right[between] * fraction[between]
    return result


def bootstrap_roc_coords(
    labels: ArrayLike,
    scores: ArrayLike,
    x: str | ArrayLike,
    *,
    input: str = "threshold",
    ret: str | tuple[str, ...] | None = None,
    direction: RocDirection = "<",
    best_method: BestMethod = "youden",
    best_weights: tuple[float, float] = (1, 0.5),
    best_policy: Literal["stop", "omit", "random"] = "stop",
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    percent: bool = False,
    stratified: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_state: int | np.random.Generator | None = None,
    bootstrap_indices: ArrayLike | None = None,
) -> BootstrapCoordsResult:
    """Bootstrap percentile intervals for fixed input values or a reselected optimum."""
    if not isinstance(confidence_level, Real) or not 0 <= confidence_level <= 1:
        raise ValueError("confidence_level must lie in [0, 1]")
    if best_policy not in ("stop", "omit", "random"):
        raise ValueError("best_policy must be 'stop', 'omit', or 'random'")
    if best_method not in ("youden", "closest.topleft"):
        raise ValueError("best_method must be 'youden' or 'closest.topleft'")
    best = isinstance(x, str)
    if best and x != "best":
        raise ValueError("bootstrap coordinates require numeric x or x='best'")
    input_name = _ALIASES.get(input, input)
    if ret is None:
        ret = _DEFAULT_RET if input_name == "threshold" else _DEFAULT_RET[1:]
    fields = _fields(ret)
    if "threshold" in fields and not (best or input_name == "threshold"):
        raise ValueError("threshold intervals require threshold input or x='best'")
    labels, scores = _binary_inputs(labels, scores)
    observed = _empirical_points(labels, scores, direction, percent)
    ratio = _weight_ratio(best_weights)
    area = _AreaSpec.create(partial_auc, focus, False, percent)
    point_estimate = _curve_coords(observed, x, input_name, fields, best_method, ratio, area)
    rng = np.random.default_rng(random_state)
    draws, count = _draws(labels == 1, n_resamples, rng, bootstrap_indices, stratified)
    values = np.full((count, 1 if best else len(point_estimate), len(fields)), np.nan)
    for i, selection in enumerate(draws):
        sample_labels = labels[selection]
        if np.all(sample_labels == sample_labels[0]):
            continue
        points = _empirical_points(sample_labels, scores[selection], observed.direction, percent)
        curves = _curve_coords(points, x, input_name, fields, best_method, ratio, area)
        if best and len(curves) != 1:
            if len(curves) < 1 or best_policy == "omit":
                continue
            if best_policy == "stop":
                raise ValueError("more than one best threshold was found in a bootstrap draw")
            curves = curves[[rng.integers(len(curves))]]
        values[i] = curves
    omitted = int(np.isnan(values).all(axis=(1, 2)).sum())
    if omitted:
        warn(f"{omitted} bootstrap draws with no usable coordinates were omitted", RuntimeWarning, stacklevel=2)
    tail = (1 - confidence_level) / 2
    probabilities = np.array([tail, 0.5, 1 - tail])
    intervals = np.apply_along_axis(_quantile_with_infinities, 0, values, probabilities).transpose(1, 2, 0)
    return BootstrapCoordsResult(
        fields, point_estimate, intervals, (~np.isnan(values)).sum(axis=0), count, omitted,
    )
