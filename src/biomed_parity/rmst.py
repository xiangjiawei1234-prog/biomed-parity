"""Two-arm restricted mean survival time inference for right-censored data."""

from dataclasses import dataclass
from math import sqrt
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy
from scipy.stats import norm

from ._survival_inputs import _fix_near_ties


@dataclass(frozen=True)
class MeanEstimate:
    estimate: float
    standard_error: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ArmEstimate:
    rmst: MeanEstimate
    rmtl: MeanEstimate


@dataclass(frozen=True)
class ContrastEstimate:
    estimate: float
    lower: float
    upper: float
    pvalue: float


@dataclass(frozen=True)
class RMSTResult:
    tau: float
    alpha: float
    arm0: ArmEstimate
    arm1: ArmEstimate
    rmst_difference: ContrastEstimate
    rmst_ratio: ContrastEstimate
    rmtl_ratio: ContrastEstimate


@dataclass(frozen=True)
class RegressionCoefficient:
    coefficient: float
    standard_error: float
    statistic: float
    pvalue: float
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class AdjustedRMSTResult:
    tau: float
    alpha: float
    rmst_difference: ContrastEstimate
    rmst_ratio: ContrastEstimate
    rmtl_ratio: ContrastEstimate
    rmst_difference_model: tuple[RegressionCoefficient, ...]
    rmst_ratio_model: tuple[RegressionCoefficient, ...]
    rmtl_ratio_model: tuple[RegressionCoefficient, ...]


@dataclass(frozen=True)
class _CensoringCurve:
    time_index: np.ndarray
    risk: np.ndarray
    censored: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_arm(cls, restricted_time: np.ndarray, known: np.ndarray) -> "_CensoringCurve":
        _, index, count = np.unique(restricted_time, return_inverse=True, return_counts=True)
        risk = len(index) + count - np.cumsum(count)
        censored = np.bincount(index, weights=~known, minlength=len(count))
        survival = np.cumprod(1 - censored / risk)[index]
        if np.any(survival <= 0):
            raise ValueError("censoring survival is zero for at least one observation")
        return cls(index, risk, censored, known / survival)

    def adjust_score(self, score: np.ndarray) -> np.ndarray:
        # Arm-specific martingale contribution to the IPCW sandwich estimator.
        totals = np.zeros((len(self.risk), score.shape[1]))
        np.add.at(totals, self.time_index, score)
        tail_score = np.cumsum(totals[::-1], axis=0)[::-1]
        scaled_tail = tail_score / self.risk[:, None]
        expected = np.cumsum(scaled_tail * (self.censored / self.risk)[:, None], axis=0)
        return (
            score + scaled_tail[self.time_index] * (self.weights == 0)[:, None]
            - expected[self.time_index]
        )


def _binary(values: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if (
        vector.ndim != 1
        or vector.dtype.kind not in "buif"
        or not np.all((vector == 0) | (vector == 1))
    ):
        raise ValueError(f"{name} must be a one-dimensional vector of 0 and 1")
    return vector.astype(bool)


def _integrate_arm(
    times: np.ndarray, events: np.ndarray, tau: float, critical: float
) -> ArmEstimate:
    # survival::survfit corrects adjacent times within floating-point tolerance.
    canonical_times = _fix_near_ties(times)

    # Only failure times strictly before tau affect the integrated curve.
    failure_times, deaths = np.unique(
        canonical_times[events & (canonical_times < tau)], return_counts=True
    )
    risk = len(times) - np.searchsorted(np.sort(canonical_times), failure_times)
    survival = np.cumprod(1 - deaths / risk)
    widths = np.diff(np.r_[0.0, failure_times, tau])
    areas = widths * np.r_[1.0, survival]
    # Sum the event-wise Greenwood terms weighted by squared remaining area.
    remaining_area = np.cumsum(areas[:0:-1])[::-1]
    mask = risk > deaths
    variance = np.sum(
        remaining_area[mask] ** 2 * (deaths[mask] / risk[mask]) / (risk[mask] - deaths[mask])
    )
    standard_error = sqrt(variance)
    rmst = float(areas.sum())
    rmtl = tau - rmst
    margin = critical * standard_error
    return ArmEstimate(
        MeanEstimate(rmst, standard_error, rmst - margin, rmst + margin),
        MeanEstimate(rmtl, standard_error, rmtl - margin, rmtl + margin),
    )


def _wald_pvalue(value: float, standard_error: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.divide(value, standard_error)
    return float(2 * norm.sf(abs(statistic)))


def _ratio(arm1: MeanEstimate, arm0: MeanEstimate, critical: float) -> ContrastEstimate:
    numerator, denominator = arm1.estimate, arm0.estimate
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = float(np.divide(numerator, denominator))
    if numerator <= 0 or denominator <= 0:
        return ContrastEstimate(estimate, np.nan, np.nan, np.nan)
    log_ratio = np.log(numerator) - np.log(denominator)
    log_se = float(np.hypot(arm1.standard_error / numerator, arm0.standard_error / denominator))
    with np.errstate(over="ignore"):
        lower, upper = np.exp(log_ratio + critical * log_se * np.array([-1, 1]))
    return ContrastEstimate(estimate, float(lower), float(upper), _wald_pvalue(log_ratio, log_se))


def _weighted_least_squares(
    design: np.ndarray, response: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    scale = np.sqrt(weights)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * scale[:, None], response * scale, rcond=None
    )
    if rank != design.shape[1]:
        raise ValueError("covariate design is rank deficient after censoring weights")
    return coefficients


def _log_link_fit(design: np.ndarray, response: np.ndarray, weights: np.ndarray) -> np.ndarray:
    active = weights > 0
    x, y, w = design[active], response[active], weights[active]
    mu = y + 0.1
    deviance = np.inf
    for _ in range(100):
        work = np.log(mu) + (y - mu) / mu
        beta = _weighted_least_squares(x, work, w * mu)
        with np.errstate(over="ignore", under="ignore"):
            mu = np.exp(x @ beta)
        if np.any(~np.isfinite(mu) | (mu <= 0)):
            break
        next_deviance = 2 * np.sum(w * (xlogy(y, y) - xlogy(y, mu) - y + mu))
        if abs(next_deviance - deviance) < 1e-10 * (0.1 + abs(next_deviance)):
            return beta
        deviance = next_deviance
    raise ValueError("weighted log-link regression failed to converge")


def _fit_adjusted(
    design: np.ndarray,
    response: np.ndarray,
    arms: np.ndarray,
    curves: tuple[_CensoringCurve, ...],
    critical: float,
    *,
    log_link: bool,
) -> tuple[RegressionCoefficient, ...]:
    weights = np.empty(len(arms))
    for mask, curve in zip((~arms, arms), curves):
        weights[mask] = curve.weights
        if log_link and not np.any(curve.weights * response[mask] > 0):
            raise ValueError("each arm must have positive weighted time for the adjusted ratios")

    if log_link:
        coefficients = _log_link_fit(design, response, weights)
        predictions = np.exp(design @ coefficients)
        derivative = predictions
    else:
        coefficients = _weighted_least_squares(design, response, weights)
        predictions = design @ coefficients
        derivative = np.ones(len(arms))

    scores = design * (weights * (response - predictions))[:, None]
    adjusted = np.concatenate([
        curve.adjust_score(scores[mask]) for mask, curve in zip((~arms, arms), curves)
    ])
    # The sandwich derivative uses the full design under the IPCW identity.
    derivative_matrix = design.T @ (design * derivative[:, None])
    influence = np.linalg.solve(derivative_matrix, adjusted.T)
    standard_errors = np.linalg.norm(influence, axis=1)

    rows = []
    for beta, se in zip(coefficients, standard_errors):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            statistic = beta / se
            interval = np.array([beta, beta - critical * se, beta + critical * se])
            estimate, lower, upper = np.exp(interval) if log_link else interval
        rows.append(RegressionCoefficient(
            float(beta), float(se), float(statistic), float(2 * norm.sf(abs(statistic))),
            float(estimate), float(lower), float(upper),
        ))
    return tuple(rows)


def _adjusted_rmst(
    times: np.ndarray,
    events: np.ndarray,
    arms: np.ndarray,
    covariates: ArrayLike,
    tau: float,
    alpha: float,
    critical: float,
) -> AdjustedRMSTResult:
    values = np.asarray(covariates)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2 or len(values) != len(times) or values.shape[1] == 0
        or values.dtype.kind not in "buif" or not np.all(np.isfinite(values))
    ):
        raise ValueError("covariates must contain finite numeric columns aligned with time")
    if np.any(times == 0) or tau == 0:
        raise ValueError("adjusted inference requires positive follow-up and tau")

    design = np.column_stack((np.ones(len(times)), arms, values))
    restricted = np.minimum(times, tau)
    known = events | (times >= tau)
    curves = tuple(
        _CensoringCurve.from_arm(restricted[mask], known[mask])
        for mask in (~arms, arms)
    )
    models = (
        _fit_adjusted(design, restricted, arms, curves, critical, log_link=False),
        _fit_adjusted(design, restricted, arms, curves, critical, log_link=True),
        _fit_adjusted(design, tau - restricted, arms, curves, critical, log_link=True),
    )
    contrasts = [
        ContrastEstimate(row.estimate, row.lower, row.upper, row.pvalue)
        for _, row, *_ in models
    ]
    return AdjustedRMSTResult(tau, alpha, *contrasts, *models)


def rmst2(
    time: ArrayLike,
    status: ArrayLike,
    arm: ArrayLike,
    *,
    tau: float | None = None,
    covariates: ArrayLike | None = None,
    alpha: float = 0.05,
) -> RMSTResult | AdjustedRMSTResult:
    """Compare two independent arms using Kaplan–Meier integrals.

    Supply finite nonnegative times and row-aligned binary status and arm
    vectors. Status 1 denotes an event and arm 0 is the reference. If tau is
    omitted, use the maximum common support permitted by the observed tails.
    Supplying covariates requests censoring-weighted regression with an identity
    link for the difference and log links for the ratios.
    """
    raw_time = np.asarray(time)
    if raw_time.ndim != 1 or raw_time.dtype.kind not in "buif":
        raise ValueError("time must be a one-dimensional vector of finite nonnegative numbers")
    times = np.asarray(raw_time, dtype=float)
    if not np.all(np.isfinite(times) & (times >= 0)):
        raise ValueError("time must be a one-dimensional vector of finite nonnegative numbers")
    events = _binary(status, "status")
    arms = _binary(arm, "arm")
    if not (len(times) == len(events) == len(arms)):
        raise ValueError("time, status and arm must have the same length")
    if not (np.any(arms) and np.any(~arms)):
        raise ValueError("arm 0 and arm 1 must each contain observations")
    if not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    grouped = [(times[mask], events[mask]) for mask in (~arms, arms)]
    maxima = [float(observed.max()) for observed, _ in grouped]
    censored_tails = [
        maximum
        for (observed, event), maximum in zip(grouped, maxima)
        if np.any(~event[observed == maximum])
    ]
    limit = min(censored_tails, default=max(maxima))
    if tau is None:
        tau = limit
    if not isinstance(tau, Real) or not 0 <= tau <= limit:
        raise ValueError(f"tau must be in [0, {limit}] for the observed follow-up")
    tau = float(tau)
    alpha = float(alpha)
    critical = float(norm.isf(alpha / 2))
    if covariates is not None:
        return _adjusted_rmst(times, events, arms, covariates, tau, alpha, critical)

    arm0, arm1 = (
        _integrate_arm(observed, event, tau, critical)
        for observed, event in grouped
    )
    difference = arm1.rmst.estimate - arm0.rmst.estimate
    se = float(np.hypot(arm1.rmst.standard_error, arm0.rmst.standard_error))
    return RMSTResult(
        tau, alpha, arm0, arm1,
        ContrastEstimate(difference, difference - critical * se, difference + critical * se,
                         _wald_pvalue(difference, se)),
        _ratio(arm1.rmst, arm0.rmst, critical),
        _ratio(arm1.rmtl, arm0.rmtl, critical),
    )
