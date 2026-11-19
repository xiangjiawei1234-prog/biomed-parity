"""Right-censored survival curve tests using the G-rho family."""

from dataclasses import dataclass
from numbers import Real
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, LinAlgWarning, solve
from scipy.stats import chi2

from ._survival_inputs import _encode_labels, _event_vector, _fix_near_ties, _numeric_vector


@dataclass(frozen=True)
class SurvDiffResult:
    """Weighted group and stratum summaries and overall test result.

    Rows of observed and expected index groups. Columns index strata when
    there are multiple strata. The variance matrix sums across strata.
    """

    groups: tuple[str | float | int, ...]
    counts: NDArray[np.int64]
    observed: NDArray[np.float64]
    expected: NDArray[np.float64]
    variance: NDArray[np.float64]
    statistic: float
    df: int
    pvalue: float
    strata: tuple[str | float | int, ...] | None
    strata_counts: NDArray[np.int64] | None


@dataclass(frozen=True)
class SurvDiffOneSampleResult:
    """Test of observed mortality against externally specified survival."""

    n: int
    observed: int
    expected: float
    variance: float
    statistic: float
    df: int
    pvalue: float


def _rho_scalar(rho: float) -> float:
    if not isinstance(rho, Real) or not np.isfinite(rho):
        raise ValueError("rho must be a finite real number")
    return float(rho)


def _stratum_statistics(time, event, group, observed, expected, variance, rho):
    n_groups = len(observed)
    risk = np.bincount(group, minlength=n_groups)
    starts = np.r_[0, np.flatnonzero(time[1:] != time[:-1]) + 1, time.size]
    survival = 1.0
    for left, right in zip(starts[:-1], starts[1:]):
        groups_at_time = group[left:right]
        events = groups_at_time[event[left:right]]
        if events.size:
            n_risk = time.size - left
            deaths = events.size
            weight = survival**rho
            proportion = risk / n_risk
            observed += weight * np.bincount(events, minlength=n_groups)
            expected += weight * deaths * proportion
            if 1 < n_risk and deaths < n_risk:
                factor = weight * weight * deaths * (n_risk - deaths) / (n_risk - 1)
                variance += factor * (np.diag(proportion) - np.outer(proportion, proportion))
            survival = survival * (n_risk - deaths) / n_risk
        risk -= np.bincount(groups_at_time, minlength=n_groups)


def survdiff(
    time: ArrayLike,
    event: ArrayLike,
    group: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    rho: float = 0,
    timefix: bool = True,
    group_order: ArrayLike | None = None,
    strata_order: ArrayLike | None = None,
) -> SurvDiffResult:
    """Compare two or more right-censored curves using a G-rho test.

    Use event indicators 0 for censored observations and 1 for events.
    Supply one group label and optionally one stratum label per observation.
    Explicit orders override ascending sorting of labels. The default time
    correction groups adjacent times within the R aeqSurv tolerance.
    """
    time = _numeric_vector(time, "time")
    event = _event_vector(event)
    if time.size != event.size:
        raise ValueError("time and event must have the same length")
    if not isinstance(timefix, (bool, np.bool_)):
        raise ValueError("timefix must be a boolean")
    rho = _rho_scalar(rho)
    groups, group_code = _encode_labels(group, group_order, time.size, "group")
    if len(groups) < 2:
        raise ValueError("there must be at least two groups")

    if strata is not None:
        strata, strata_code = _encode_labels(strata, strata_order, time.size, "strata")
        strata_counts = np.bincount(strata_code).astype(np.int64, copy=False)
        boundaries = np.r_[0, np.cumsum(strata_counts)]
    else:
        if strata_order is not None:
            raise ValueError("strata_order requires strata")
        strata_code = np.zeros(time.size, dtype=np.intp)
        strata_counts = None
        boundaries = (0, time.size)

    if timefix:
        time = _fix_near_ties(time)
    order = np.lexsort((time, strata_code))
    n_groups = len(groups)
    n_strata = len(boundaries) - 1
    observed = np.zeros((n_groups, n_strata))
    expected = np.zeros_like(observed)
    variance = np.zeros((n_groups, n_groups))
    for j, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        rows = order[left:right]
        _stratum_statistics(
            time[rows], event[rows], group_code[rows],
            observed[:, j], expected[:, j], variance, rho,
        )

    fitted = expected.sum(axis=1)
    contributing = np.flatnonzero(fitted > 0)
    df = contributing.size - 1
    statistic = 0.0
    if df > 0:
        # One contrast is redundant because group residuals sum to zero.
        contrasts = contributing[1:]
        residual = observed.sum(axis=1)[contrasts] - fitted[contrasts]
        covariance = variance[np.ix_(contrasts, contrasts)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            try:
                statistic = float(residual @ solve(covariance, residual, assume_a="gen"))
            except (LinAlgError, LinAlgWarning) as error:
                raise LinAlgError("contributing groups have a singular covariance") from error
    pvalue = float(chi2.sf(statistic, df)) if df != 0 else 1.0
    if n_strata == 1:
        observed = observed[:, 0]
        expected = expected[:, 0]

    return SurvDiffResult(
        groups=groups,
        counts=np.bincount(group_code).astype(np.int64, copy=False),
        observed=observed,
        expected=expected,
        variance=variance,
        statistic=statistic,
        df=df,
        pvalue=pvalue,
        strata=strata,
        strata_counts=strata_counts,
    )


def survdiff_one_sample(
    event: ArrayLike,
    survival_probability: ArrayLike,
    *,
    rho: float = 0,
) -> SurvDiffOneSampleResult:
    """Test a right-censored sample against survival probabilities at follow-up.

    Each probability is supplied at the corresponding subject's observed
    event or censoring time. Fitting or estimating those probabilities is
    outside this test.
    """
    event = _event_vector(event)
    probability = _numeric_vector(survival_probability, "survival_probability")
    rho = _rho_scalar(rho)
    if event.size != probability.size:
        raise ValueError("event and survival_probability must have the same length")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("survival_probability must be in [0, 1]")

    observed = int(event.sum())
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expected = np.sum(-np.log(probability))
        if rho == 0:
            variance = expected
            score = observed - expected
        else:
            score = np.sum(1 / rho - (1 / rho + event) * probability**rho)
            variance = np.sum(1 - probability ** (2 * rho)) / (2 * rho)
        statistic = float(np.divide(score * score, variance))
    return SurvDiffOneSampleResult(
        n=event.size,
        observed=observed,
        expected=float(expected),
        variance=float(variance),
        statistic=statistic,
        df=1,
        pvalue=float(chi2.sf(statistic, 1)),
    )
