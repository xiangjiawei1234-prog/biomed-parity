"""Intercept-only random-effects meta-analysis with REML."""

from dataclasses import dataclass
from math import copysign, hypot, inf, nan, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import betaincinv
from scipy.stats import chi2, norm, t


@dataclass(frozen=True)
class REMLMetaAnalysisResult:
    study_count: int
    mean_effect: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    prediction_interval: tuple[float, float]
    tau2: float
    tau2_confidence_interval: tuple[float, float]
    tau2_upper_limit_reached: bool
    i2_percent: float
    i2_confidence_interval: tuple[float, float]
    h2: float
    h2_confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class _Profile:
    y: NDArray[np.float64]
    v: NDArray[np.float64]

    def terms(self, tau2: float):
        total = self.v + tau2
        smallest = float(total.min())
        weights = smallest / total
        weight_sum = float(weights.sum())
        mean = float(np.dot(weights / weight_sum, self.y))
        return smallest, weights, weight_sum, mean, self.y - mean

    def objective(self, tau2: float) -> float:
        """Negative twice the restricted log-likelihood, up to a constant."""
        smallest, weights, weight_sum, _, residuals = self.terms(tau2)
        return float(
            np.log(self.v + tau2).sum() + np.log(weight_sum) - np.log(smallest)
            + np.dot(weights, residuals**2) / smallest
        )

    def score(self, tau2: float) -> float:
        smallest, weights, weight_sum, _, residuals = self.terms(tau2)
        # Scaled by 2 * smallest**2 to preserve the sign without large weights.
        return float(
            np.dot(weights**2, residuals**2) - smallest * _trace(weights, weight_sum)
        )


def _trace(weights: NDArray[np.float64], weight_sum: float) -> float:
    # Equivalent to sum(w) - sum(w**2)/sum(w), without cancellation.
    return float(2 * np.dot(weights[1:], np.cumsum(weights[:-1])) / weight_sum)


def _fit_tau2(profile: _Profile) -> float:
    ceiling = float(np.ptp(profile.y) ** 2)
    if ceiling == 0:
        return 0.0

    # At tau² >= (max(y) - min(y))² the restricted score is nonpositive.
    # Compare interior maxima with the boundary over the relevant scales.
    smallest = min(ceiling, max(np.finfo(float).tiny, float(profile.v.min()) * 1e-6))
    steps = max(32, int(12 * (np.log10(ceiling) - np.log10(smallest))))
    grid = np.concatenate(([0.0], np.geomspace(smallest, ceiling, steps)))
    scores = [profile.score(float(point)) for point in grid]
    candidates = [0.0, ceiling]
    for left, right, score_left, score_right in zip(
        grid[:-1], grid[1:], scores[:-1], scores[1:], strict=True
    ):
        if score_left > 0 and score_right <= 0:
            candidates.append(brentq(profile.score, left, right, xtol=np.finfo(float).tiny))
    return min(candidates, key=profile.objective)


def _profile_interval(
    profile: _Profile, tau2: float, confidence_level: float, upper_limit: float
) -> tuple[tuple[float, float], bool]:
    cutoff = profile.objective(tau2) + float(chi2.isf(1 - confidence_level, df=1))

    def target(value):
        return profile.objective(value) - cutoff

    lower = brentq(target, 0, tau2, xtol=np.finfo(float).tiny) if target(0) > 0 else 0.0
    upper = min(upper_limit, max(2 * tau2, float(profile.v.max())))
    while upper < upper_limit and target(upper) < 0:
        upper = min(2 * upper, upper_limit)
    if target(upper) < 0:
        return (lower, upper), True
    return (lower, brentq(target, tau2, upper, xtol=np.finfo(float).tiny)), False


def _student_critical(level: float, degrees: int) -> float:
    # Invert the appropriate beta tail to preserve accuracy near 0 and 1.
    if level <= 0.5:
        squared_ratio = float(betaincinv(0.5, degrees / 2, level))
        return sqrt(degrees * squared_ratio / (1 - squared_ratio))
    complement = float(betaincinv(degrees / 2, 0.5, 1 - level))
    return sqrt(degrees * (1 - complement) / complement)


def meta_reml(
    effects: ArrayLike,
    variances: ArrayLike,
    *,
    test: Literal["z", "t", "knha", "hksj", "adhoc"] = "z",
    prediction: Literal["default", "riley"] = "default",
    confidence_level: float = 0.95,
    tau2_upper_limit: float | None = None,
) -> REMLMetaAnalysisResult:
    """Pool independent study effects with known, strictly positive variances.

    The true-effect prediction interval and inference follow metafor's
    intercept-only REML model. Heterogeneity intervals use profile likelihood.
    """
    if test not in ("z", "t", "knha", "hksj", "adhoc"):
        raise ValueError("test must be 'z', 't', 'knha', 'hksj', or 'adhoc'")
    if prediction not in ("default", "riley"):
        raise ValueError("prediction must be 'default' or 'riley'")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    effects = np.asarray(effects)
    variances = np.asarray(variances)
    k = effects.size
    if effects.ndim != 1 or effects.shape != variances.shape or k < 2:
        raise ValueError("effects and variances must be equal-length 1D arrays with at least two studies")
    if effects.dtype.kind not in "iuf" or not np.isfinite(effects).all():
        raise ValueError("effects must be finite real numbers")
    if variances.dtype.kind not in "iuf" or not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("variances must be finite and strictly positive real numbers")
    effects = effects.astype(float, copy=False)
    variances = variances.astype(float, copy=False)

    center = float(effects[0])
    with np.errstate(over="ignore", under="ignore"):
        centered = effects - center
        scale = max(float(np.max(np.abs(centered)) ** 2), float(variances.max()))
        scaled_variances = variances / scale
    if not np.isfinite(scale) or scaled_variances.min() < np.finfo(float).tiny:
        raise ValueError("effects and variances exceed the supported numerical range")

    std_scale = sqrt(scale)
    profile = _Profile(centered / std_scale, scaled_variances)
    scaled_tau2 = _fit_tau2(profile)
    smallest, weights, weight_sum, mean, residuals = profile.terms(scaled_tau2)
    tau2 = scaled_tau2 * scale
    mean_effect = mean * std_scale + center
    variance_of_mean = smallest / weight_sum
    if test in ("knha", "hksj", "adhoc"):
        # HK multiplies the inverse-weight variance by the residual mean square.
        hk_scale = float(np.dot(weights, residuals**2)) / ((k - 1) * smallest)
        variance_of_mean *= max(1.0, hk_scale) if test == "adhoc" else hk_scale
    standard_error = sqrt(variance_of_mean) * std_scale
    distribution = norm if test == "z" else t(df=k - 1)
    if standard_error:
        statistic = mean_effect / standard_error
    else:
        statistic = copysign(inf, mean_effect) if mean_effect else nan
    p_value = float(2 * distribution.sf(abs(statistic)))
    tail = (1 - confidence_level) / 2
    critical = float(norm.isf(tail)) if test == "z" else _student_critical(confidence_level, k - 1)
    ci_half = critical * standard_error
    prediction_critical = (
        _student_critical(confidence_level, max(1, k - 2)) if prediction == "riley" else critical
    )
    pi_half = prediction_critical * hypot(standard_error, sqrt(tau2))

    if tau2_upper_limit is None:
        tau2_upper_limit = max(100.0, 10 * tau2)
    if not np.isfinite(tau2_upper_limit) or tau2_upper_limit <= tau2:
        raise ValueError("tau2_upper_limit must be finite and greater than the fitted tau2")
    scaled_interval, limit_reached = _profile_interval(
        profile, scaled_tau2, confidence_level, tau2_upper_limit / scale
    )
    tau2_lower, tau2_upper = (endpoint * scale for endpoint in scaled_interval)
    smallest_fixed, fixed_weights, fixed_sum, _, _ = profile.terms(0.0)
    typical_variance = (k - 1) * smallest_fixed / _trace(fixed_weights, fixed_sum)

    def i2(value):
        return 100 * (value / (value + typical_variance))

    def h2(value):
        return 1 + value / typical_variance

    return REMLMetaAnalysisResult(
        study_count=k,
        mean_effect=mean_effect,
        standard_error=standard_error,
        statistic=statistic,
        p_value=p_value,
        confidence_interval=(mean_effect - ci_half, mean_effect + ci_half),
        prediction_interval=(mean_effect - pi_half, mean_effect + pi_half),
        tau2=tau2,
        tau2_confidence_interval=(tau2_lower, tau2_upper),
        tau2_upper_limit_reached=limit_reached,
        i2_percent=i2(scaled_tau2),
        i2_confidence_interval=(i2(scaled_interval[0]), i2(scaled_interval[1])),
        h2=h2(scaled_tau2),
        h2_confidence_interval=(h2(scaled_interval[0]), h2(scaled_interval[1])),
    )
