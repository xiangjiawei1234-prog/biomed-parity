"""DeLong inference for full empirical binary ROC AUCs."""

from dataclasses import dataclass
from math import copysign, inf, nan, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, t

Direction = Literal["<", ">"]
Alternative = Literal["two.sided", "less", "greater"]


@dataclass(frozen=True)
class DeLongAucResult:
    auc: float
    variance: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class DeLongPairedResult:
    auc_1: float
    auc_2: float
    variance_1: float
    variance_2: float
    covariance: float
    difference: float
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class DeLongUnpairedResult:
    auc_1: float
    auc_2: float
    variance_1: float
    variance_2: float
    difference: float
    statistic: float
    degrees_of_freedom: float
    p_value: float


@dataclass(frozen=True)
class _Placements:
    # Twice the number of wins, so tied cross-class comparisons count exactly.
    case_wins: NDArray[np.intp]
    control_wins: NDArray[np.intp]

    @property
    def numerator(self) -> int:
        return int(self.case_wins.sum())

    @property
    def denominator(self) -> int:
        return 2 * self.case_wins.size * self.control_wins.size

    @property
    def auc(self) -> float:
        return self.numerator / self.denominator

    @property
    def variance(self) -> float:
        return _variance_of_auc(self.case_wins, self.control_wins)


def _variance_of_auc(case_wins: NDArray[np.intp], control_wins: NDArray[np.intp]) -> float:
    n_cases, n_controls = case_wins.size, control_wins.size
    from_cases = case_wins.var(ddof=1) / (n_cases * (2 * n_controls) ** 2)
    from_controls = control_wins.var(ddof=1) / (n_controls * (2 * n_cases) ** 2)
    return float(from_cases + from_controls)


def _covariance_of_aucs(first: _Placements, second: _Placements) -> float:
    n_cases, n_controls = first.case_wins.size, first.control_wins.size
    from_cases = np.cov(first.case_wins, second.case_wins)[0, 1]
    from_controls = np.cov(first.control_wins, second.control_wins)[0, 1]
    return float(
        from_cases / (n_cases * (2 * n_controls) ** 2)
        + from_controls / (n_controls * (2 * n_cases) ** 2)
    )


def _binary_inputs(labels: ArrayLike, scores: ArrayLike):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be one-dimensional arrays of equal size")
    if labels.dtype.kind not in "biuf" or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 for controls and 1 for cases")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("scores must be finite real numbers")
    return labels, scores


def _placements(labels: ArrayLike, scores: ArrayLike, direction: Direction) -> _Placements:
    if direction not in ("<", ">"):
        raise ValueError("direction must be '<' or '>'")
    labels, scores = _binary_inputs(labels, scores)
    cases = scores[labels == 1]
    controls = scores[labels == 0]
    if cases.size < 2 or controls.size < 2:
        raise ValueError("DeLong variance requires at least two cases and two controls")

    # Binary search yields the same placements as all pairwise comparisons,
    # while preserving subject order for the paired covariance calculation.
    sorted_controls, sorted_cases = np.sort(controls), np.sort(cases)
    control_ranks = np.searchsorted(sorted_controls, cases, side="left")
    control_ranks += np.searchsorted(sorted_controls, cases, side="right")
    case_ranks = np.searchsorted(sorted_cases, controls, side="left")
    case_ranks += np.searchsorted(sorted_cases, controls, side="right")

    if direction == "<":
        return _Placements(control_ranks, 2 * cases.size - case_ranks)
    return _Placements(2 * controls.size - control_ranks, case_ranks)


def _confidence_critical_value(confidence_level: float) -> float:
    if not 0 <= confidence_level <= 1:
        raise ValueError("confidence_level must lie in [0, 1]")
    return float(norm.isf((1 - confidence_level) / 2))


def _validate_alternative(alternative: Alternative) -> None:
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be 'two.sided', 'less', or 'greater'")


def _p_value(statistic: float, distribution, alternative: Alternative) -> float:
    if alternative == "less":
        return float(distribution.cdf(statistic))
    if alternative == "greater":
        return float(distribution.sf(statistic))
    return float(2 * distribution.sf(abs(statistic)))


def delong_auc(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    direction: Direction = "<",
    confidence_level: float = 0.95,
) -> DeLongAucResult:
    """Estimate full binary AUC and a two-sided DeLong interval clipped to [0, 1].

    Labels 0 and 1 indicate controls and cases. Direction '<' means higher
    scores indicate cases and '>' means lower scores indicate cases.
    """
    critical = _confidence_critical_value(confidence_level)
    placements = _placements(labels, scores, direction)
    auc, variance = placements.auc, placements.variance
    if confidence_level == 1:
        interval = (0.0, 1.0)
    else:
        margin = critical * sqrt(variance)
        interval = (max(0.0, auc - margin), min(1.0, auc + margin))
    return DeLongAucResult(auc, variance, interval)


def delong_paired_test(
    labels: ArrayLike,
    scores_1: ArrayLike,
    scores_2: ArrayLike,
    *,
    direction: Direction = "<",
    alternative: Alternative = "two.sided",
    confidence_level: float = 0.95,
) -> DeLongPairedResult:
    """Compare two full AUCs scored on the same subjects in the same order.

    Difference and statistic use AUC 1 minus AUC 2. The confidence interval
    for the difference is two-sided for every alternative hypothesis.
    """
    _validate_alternative(alternative)
    critical = _confidence_critical_value(confidence_level)
    first = _placements(labels, scores_1, direction)
    second = _placements(labels, scores_2, direction)
    difference = (first.numerator - second.numerator) / first.denominator

    # The variance of differences avoids cancellation for highly correlated curves.
    variance = _variance_of_auc(
        first.case_wins - second.case_wins, first.control_wins - second.control_wins
    )
    standard_error = sqrt(variance)
    if standard_error:
        statistic = difference / standard_error
    else:
        statistic = copysign(inf, difference) if difference else 0.0
    margin = critical * standard_error

    return DeLongPairedResult(
        auc_1=first.auc,
        auc_2=second.auc,
        variance_1=first.variance,
        variance_2=second.variance,
        covariance=_covariance_of_aucs(first, second),
        difference=difference,
        statistic=statistic,
        p_value=_p_value(statistic, norm, alternative),
        confidence_interval=(difference - margin, difference + margin),
    )


def delong_unpaired_test(
    labels_1: ArrayLike,
    scores_1: ArrayLike,
    labels_2: ArrayLike,
    scores_2: ArrayLike,
    *,
    direction: Direction = "<",
    alternative: Alternative = "two.sided",
) -> DeLongUnpairedResult:
    """Compare two full AUCs from independent cohorts using pROC's t reference."""
    _validate_alternative(alternative)
    first = _placements(labels_1, scores_1, direction)
    second = _placements(labels_2, scores_2, direction)
    variance_1, variance_2 = first.variance, second.variance
    difference = first.auc - second.auc
    total_variance = variance_1 + variance_2

    if total_variance == 0:
        statistic = copysign(inf, difference) if difference else nan
        degrees_of_freedom, p_value = nan, nan
    else:
        statistic = difference / sqrt(total_variance)
        degrees_of_freedom = total_variance**2 / (
            variance_1**2 / (first.case_wins.size + first.control_wins.size - 1)
            + variance_2**2 / (second.case_wins.size + second.control_wins.size - 1)
        )
        p_value = _p_value(statistic, t(df=degrees_of_freedom), alternative)

    return DeLongUnpairedResult(
        auc_1=first.auc,
        auc_2=second.auc,
        variance_1=variance_1,
        variance_2=variance_2,
        difference=difference,
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
    )
