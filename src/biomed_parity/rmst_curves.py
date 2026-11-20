"""Plot-ready Kaplan–Meier steps and restricted-mean regions for two arms."""

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ._survival_inputs import _fix_near_ties
from .rmst import RMSTResult, rmst2


@dataclass(frozen=True)
class RMSTArmCurve:
    """Observed KM knots, full curve steps, and area polygons through tau."""

    time: NDArray[np.float64]
    survival: NDArray[np.float64]
    n_risk: NDArray[np.intp]
    n_event: NDArray[np.intp]
    n_censor: NDArray[np.intp]
    log_survival_standard_error: NDArray[np.float64]
    steps: NDArray[np.float64]
    rmst_polygons: NDArray[np.float64]
    rmtl_polygons: NDArray[np.float64]
    confidence_band: NDArray[np.float64] | None


@dataclass(frozen=True)
class RMSTCurveData:
    """Unadjusted RMST inference and two arms of numerical plot data."""

    summary: RMSTResult
    arm0: RMSTArmCurve
    arm1: RMSTArmCurve
    pointwise_confidence_level: float | None

    def to_dict(self) -> dict:
        """Return JSON-compatible plot data with nonfinite values represented by None."""
        return _json_values(asdict(self))


def _json_values(value):
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, list):
        return [_json_values(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _step_values(values):
    return np.repeat(np.r_[1.0, values], 2)[:-1]


def _area_polygons(time, survival, tau):
    before_tau = time <= tau
    breaks = np.r_[0.0, time[before_tau], tau]
    keep = np.diff(breaks) > 0
    left, right = breaks[:-1][keep], breaks[1:][keep]
    height = np.r_[1.0, survival[before_tau]][keep]
    x = np.column_stack((left, right, right, left))
    zeros, ones = np.zeros_like(height), np.ones_like(height)
    below = np.column_stack((zeros, zeros, height, height))
    above = np.column_stack((height, height, ones, ones))
    return np.stack((x, below), axis=-1), np.stack((x, above), axis=-1)


def _fit_arm(time, event, tau, confidence_critical):
    knots, index, counts = np.unique(_fix_near_ties(time), return_inverse=True, return_counts=True)
    risk = int(time.size) - np.r_[0, np.cumsum(counts[:-1])]
    deaths = np.bincount(index[event], minlength=len(knots))
    censored = counts - deaths
    survival = np.cumprod(1 - deaths / risk)
    with np.errstate(divide="ignore"):
        log_se = np.sqrt(np.cumsum((deaths / risk) / (risk - deaths)))

    step_time = np.r_[0.0, np.repeat(knots, 2)]
    steps = np.column_stack((step_time, _step_values(survival)))
    rmst, rmtl = _area_polygons(knots, survival, tau)
    band = None
    if confidence_critical is not None:
        lower, upper = np.full_like(survival, np.nan), np.full_like(survival, np.nan)
        positive = survival > 0
        log_survival = np.log(survival[positive])
        margin = confidence_critical * log_se[positive]
        lower[positive] = np.exp(log_survival - margin)
        upper[positive] = np.exp(np.minimum(0, log_survival + margin))
        band = np.column_stack((step_time, _step_values(lower), _step_values(upper)))
    return RMSTArmCurve(knots, survival, risk, deaths, censored, log_se, steps, rmst, rmtl, band)


def rmst_curve_data(
    time: ArrayLike,
    status: ArrayLike,
    arm: ArrayLike,
    *,
    tau: float | None = None,
    alpha: float = 0.05,
    pointwise_confidence_level: float | None = None,
) -> RMSTCurveData:
    """Return unadjusted RMST, two full KM curves, and polygons ending at tau.

    Polygon arrays have one row per nonzero-width time interval and four
    (time, survival) vertices per row. A requested confidence band has
    (time, lower, upper) columns and uses pointwise Greenwood log limits.
    """
    confidence_critical = None
    if pointwise_confidence_level is not None:
        if not isinstance(pointwise_confidence_level, Real) or not 0 < pointwise_confidence_level < 1:
            raise ValueError("pointwise_confidence_level must be between 0 and 1")
        pointwise_confidence_level = float(pointwise_confidence_level)
        confidence_critical = float(norm.isf((1 - pointwise_confidence_level) / 2))
    summary = rmst2(time, status, arm, tau=tau, alpha=alpha)
    times, events, arms = np.asarray(time, dtype=float), np.asarray(status, dtype=bool), np.asarray(arm, dtype=bool)
    arm0 = _fit_arm(times[~arms], events[~arms], summary.tau, confidence_critical)
    arm1 = _fit_arm(times[arms], events[arms], summary.tau, confidence_critical)
    return RMSTCurveData(summary, arm0, arm1, pointwise_confidence_level)
