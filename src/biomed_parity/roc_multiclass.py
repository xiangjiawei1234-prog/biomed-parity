"""Hand–Till multiclass ROC areas from single scores or class-score matrices."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from .roc_bootstrap import Focus, RocDirection, _AreaSpec, _EmpiricalROC
from .roc_thresholds import _resolve_direction

ClassLabel = str | int | float


@dataclass(frozen=True)
class ClassPairAUC:
    """AUC and binary components for one unordered pair of classes."""

    classes: tuple[ClassLabel, ClassLabel]
    auc: float
    component_aucs: tuple[float, ...]
    directions: tuple[RocDirection, ...]


@dataclass(frozen=True)
class MulticlassAUCResult:
    """Macro-average AUC, included class levels, and pairwise components."""

    auc: float
    levels: tuple[ClassLabel, ...]
    pairs: tuple[ClassPairAUC, ...]


def _present_labels(values, name):
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    present = np.array([
        label is not None and not (isinstance(label, (float, np.floating)) and np.isnan(label))
        for label in labels
    ], dtype=bool)
    if not (all(isinstance(label, str) for label in labels[present]) or all(
        isinstance(label, Real) and np.isfinite(float(label)) for label in labels[present]
    )):
        raise ValueError(f"{name} must contain strings or finite real numbers of one type")
    return labels, present


def _declared_levels(values, name):
    labels, present = _present_labels(values, name)
    declared = tuple(labels.tolist())
    if not present.all() or len(set(declared)) != len(declared):
        raise ValueError(f"{name} must contain unique nonmissing class labels")
    return declared


def _class_columns(levels, score_labels, count):
    if score_labels is None:
        raise ValueError("score_labels must name each column of a class-score matrix")
    columns = _declared_levels(score_labels, "score_labels")
    if len(columns) != count:
        raise ValueError("score_labels must have one label per score column")
    selected = {name: i for i, name in enumerate(columns) if name in levels}
    if len(selected) < 2:
        raise ValueError("at least two class-score columns must match the selected levels")
    missing = [name for name in levels if name not in selected]
    extra = [name for name in columns if name not in levels]
    if missing:
        warn(f"no score columns for selected classes {missing!r}", RuntimeWarning, stacklevel=3)
    if extra:
        warn(f"ignoring score columns outside the selected classes {extra!r}", RuntimeWarning, stacklevel=3)
    return selected


def _pair_auc(controls, cases, direction, na_rm, area, allow_below_chance):
    if na_rm:
        controls, cases = controls[~np.isnan(controls)], cases[~np.isnan(cases)]
    elif np.isnan(controls).any() or np.isnan(cases).any():
        raise ValueError("scores in each comparison must be present when na_rm=False")
    if controls.size == 0 or cases.size == 0:
        raise ValueError("each ROC comparison requires a usable score in both classes")
    scores = np.concatenate((controls, cases))
    labels = np.r_[np.zeros(controls.size, dtype=bool), np.ones(cases.size, dtype=bool)]
    direction = _resolve_direction(labels, scores, direction)
    roc = _EmpiricalROC.create(labels, scores, direction)
    return area.auc(roc, allow_below_chance=allow_below_chance), direction


def multiclass_auc(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    levels: ArrayLike | None = None,
    score_labels: ArrayLike | None = None,
    direction: RocDirection | None = None,
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
    na_rm: bool = True,
) -> MulticlassAUCResult:
    """Compute the mean AUC across class pairs from shared or class-specific scores.

    By default, a vector uses pairwise median-based direction and a matrix
    uses higher scores for the named class. Matrix columns require score_labels.
    Missing scores are removed separately for each binary comparison by default.
    """
    labels, present = _present_labels(labels, "labels")
    scores = np.asarray(scores)
    if scores.ndim not in (1, 2) or scores.shape[0] != labels.size:
        raise ValueError("scores must be a vector or matrix with one row per label")
    if scores.dtype.kind not in "iuf" or np.isinf(scores).any():
        raise ValueError("scores must be real numbers without infinity")
    if not na_rm and not present.all():
        raise ValueError("labels must be present when na_rm=False")
    labels, scores = labels[present], scores[present]

    observed = set(labels.tolist())
    levels = tuple(sorted(observed)) if levels is None else _declared_levels(levels, "levels")
    if len(levels) < 2 or not set(levels) <= observed:
        raise ValueError("levels must select at least two observed classes")
    if direction not in (None, "auto", "<", ">"):
        raise ValueError("direction must be '<', '>', or 'auto'")

    multivariate = scores.ndim == 2
    if multivariate:
        if direction == "auto":
            raise ValueError("direction='auto' is only available for single-score multiclass ROC")
        direction = ">" if direction is None else direction
        columns = _class_columns(levels, score_labels, scores.shape[1])
        levels = tuple(columns)
    else:
        if score_labels is not None:
            raise ValueError("score_labels is only used for class-score matrices")
        direction = "auto" if direction is None else direction

    area = _AreaSpec.create(partial_auc, focus, standardize, percent)
    groups = {label: scores[labels == label] for label in levels}
    pairs = []
    for first, second in combinations(levels, 2):
        first_rows, second_rows = groups[first], groups[second]
        comparisons = ((first_rows, second_rows),)
        if multivariate:
            comparisons = (
                (first_rows[:, columns[first]], second_rows[:, columns[first]]),
                (second_rows[:, columns[second]], first_rows[:, columns[second]]),
            )
        values, directions = zip(*(
            _pair_auc(controls, cases, direction, na_rm, area, allow_below_chance)
            for controls, cases in comparisons
        ))
        pairs.append(ClassPairAUC((first, second), sum(values) / len(values), values, directions))
    return MulticlassAUCResult(sum(pair.auc for pair in pairs) / len(pairs), levels, tuple(pairs))
