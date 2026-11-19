"""Bootstrap internal validation of a binary logistic prediction model."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata

_METRICS = ("Dxy", "R2", "Intercept", "Slope", "Emax", "D", "U", "Q", "B", "g", "gp")


@dataclass(frozen=True)
class LogisticValidationResult:
    """One vector per output column, indexed by metric_names."""

    metric_names: tuple[str, ...]
    original: NDArray[np.float64]
    training: NDArray[np.float64]
    test: NDArray[np.float64]
    optimism: NDArray[np.float64]
    corrected: NDArray[np.float64]
    lower: NDArray[np.float64] | None
    upper: NDArray[np.float64] | None
    counts: NDArray[np.int64]
    failed_replicates: tuple[int, ...]


def _finite_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _deviance(signed, lp):
    return 2 * np.logaddexp(0, -signed * lp).sum()


def _logistic_fit(x, y):
    if y.min() == y.max():
        raise np.linalg.LinAlgError("the sample must contain both response classes")
    center = x.mean(axis=0)
    spread = x.std(axis=0)
    if np.any(spread == 0):
        raise np.linalg.LinAlgError("a predictor is constant in this sample")
    design = np.column_stack((np.ones(len(y)), (x - center) / spread))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise np.linalg.LinAlgError("the logistic design matrix is rank deficient")
    signed = 2 * y - 1
    coefficients = np.r_[logit(y.mean()), np.zeros(x.shape[1])]
    lp = design @ coefficients
    deviance = _deviance(signed, lp)
    for _ in range(80):
        weights = expit(lp) * expit(-lp)
        score = design.T @ (signed * expit(-signed * lp))
        information = design.T @ (weights[:, None] * design)
        delta = np.linalg.solve(information, score)
        if np.max(np.abs(delta)) <= 1e-9:
            slopes = coefficients[1:] / spread
            return np.r_[coefficients[0] - center @ slopes, slopes]
        for half in range(25):
            updated = coefficients + delta * 0.5**half
            fitted = design @ updated
            candidate = _deviance(signed, fitted)
            if candidate <= deviance + np.finfo(float).eps * deviance:
                coefficients, lp, deviance = updated, fitted, candidate
                break
        else:
            break
    raise np.linalg.LinAlgError("logistic fit did not converge to a finite maximum")


def _gini_difference(values):
    n = len(values)
    rank_weights = 2 * np.arange(n) + 1 - n
    return 2 * np.sum(rank_weights * np.sort(values - values.mean())) / (n * (n - 1))


def _performance(y, lp, emax_grid, *, apparent):
    signed = 2 * y - 1
    n = len(y)
    probabilities = expit(lp)
    brier = np.mean((y - probabilities) ** 2)
    rate = y.mean()
    null_deviance = -2 * n * (rate * np.log(rate) + (1 - rate) * np.log1p(-rate))
    if apparent:
        intercept, slope, emax = 0.0, 1.0, 0.0
        calibrated = lp
        deviance = _deviance(signed, calibrated)
        unreliability = -2 / n
    else:
        intercept, slope = _logistic_fit(lp[:, None], y)
        calibrated = intercept + slope * lp
        deviance = _deviance(signed, calibrated)
        unreliability = (_deviance(signed, np.clip(lp, -40, 40)) - deviance - 2) / n
        with np.errstate(invalid="ignore"):
            calibrated_grid = expit(intercept + slope * logit(emax_grid))
        emax = np.nanmax(np.abs(calibrated_grid - emax_grid))

    positive = y == 1
    n_positive = positive.sum()
    auc = (rankdata(calibrated)[positive].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * (n - n_positive)
    )
    lr = null_deviance - deviance
    r2 = np.expm1(-lr / n) / np.expm1(-null_deviance / n)
    discrimination = (lr - 1) / n
    return np.array([
        2 * auc - 1, r2, intercept, slope, emax, discrimination, unreliability,
        discrimination - unreliability, brier, _gini_difference(calibrated),
        _gini_difference(expit(calibrated)),
    ])


def _resamples(n, supplied, count, seed):
    if supplied is None:
        count = 40 if count is None else count
        if not isinstance(count, Integral) or count < 1:
            raise ValueError("n_resamples must be a positive integer")
        return np.random.default_rng(seed).integers(0, n, (count, n))
    if count is not None or seed is not None:
        raise ValueError("n_resamples and seed apply when bootstrap_indices is omitted")
    indices = np.asarray(supplied)
    if indices.ndim != 2 or indices.shape[1] != n or indices.shape[0] == 0:
        raise ValueError("bootstrap_indices must have shape (replicates, observations)")
    if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("bootstrap_indices must contain zero-based integer row indices")
    return indices


def _intervals(training, test, corrected, confidence):
    if confidence is None:
        return None, None
    values = training - 1.25 * test
    critical = ndtri((1 + confidence) / 2)
    if len(values) < 2:
        missing = np.full(len(_METRICS), np.nan)
        return missing.copy(), missing
    if len(values) < 10:
        top = bottom = np.std(values, axis=0, ddof=1)
    else:
        centered = values - values.mean(axis=0)
        top_mask = centered >= 0
        bottom_mask = centered <= 0
        with np.errstate(invalid="ignore", divide="ignore"):
            top = np.sqrt(np.sum(centered**2 * top_mask, axis=0) / (top_mask.sum(axis=0) - 1))
            bottom = np.sqrt(
                np.sum(centered**2 * bottom_mask, axis=0) / (bottom_mask.sum(axis=0) - 1)
            )
    return corrected - critical * top, corrected + critical * bottom


def validate_logistic(
    x: ArrayLike,
    y: ArrayLike,
    *,
    bootstrap_indices: ArrayLike | None = None,
    n_resamples: int | None = None,
    seed: int | np.random.Generator | None = None,
    emax_limits: tuple[float, float] = (0, 1),
    confidence: float | None = 0.95,
) -> LogisticValidationResult:
    """Estimate bootstrap optimism and corrected binary logistic performance.

    Supply a finite numeric design matrix without an intercept, a 0/1
    outcome, and optionally a matrix of zero-based row indices. Each
    replicate is fitted on its sampled rows and evaluated on both that
    sample and on the original data. Failed fits are counted separately.
    """
    x = _finite_array(x, "x")
    y = _finite_array(y, "y")
    if x.ndim != 2 or x.shape[1] < 1 or y.ndim != 1 or x.shape[0] != len(y) or not len(y):
        raise ValueError("x must be a nonempty design matrix with one row per y value")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("y must contain both binary outcome classes encoded as 0 and 1")
    limits = _finite_array(emax_limits, "emax_limits")
    if limits.shape != (2,) or not 0 <= limits[0] < limits[1] <= 1:
        raise ValueError("emax_limits must be an increasing pair in [0, 1]")
    if confidence is not None and (not isinstance(confidence, Real) or not 0 < confidence < 1):
        raise ValueError("confidence must be None or in (0, 1)")

    low, high = limits
    emax_grid = np.minimum(low + 0.0005 * np.arange(1 + int((high - low) / 0.0005 + 1e-10)), high)
    indices = _resamples(len(y), bootstrap_indices, n_resamples, seed)
    original_coefficients = _logistic_fit(x, y)
    original_lp = original_coefficients[0] + x @ original_coefficients[1:]
    original = _performance(y, original_lp, emax_grid, apparent=True)
    training = []
    test = []
    failures = []
    for i, selection in enumerate(indices):
        try:
            coefficients = _logistic_fit(x[selection], y[selection])
            lp = coefficients[0] + x @ coefficients[1:]
            train_metrics = _performance(y[selection], lp[selection], emax_grid, apparent=True)
            test_metrics = _performance(y, lp, emax_grid, apparent=False)
        except np.linalg.LinAlgError:
            failures.append(i)
            continue
        training.append(train_metrics)
        test.append(test_metrics)
    if not training:
        raise ValueError("no successful bootstrap fits")
    training = np.asarray(training)
    test = np.asarray(test)
    training_mean = training.mean(axis=0)
    test_mean = test.mean(axis=0)
    optimism = training_mean - test_mean
    corrected = original - optimism
    lower, upper = _intervals(training, test, corrected, confidence)
    return LogisticValidationResult(
        metric_names=_METRICS,
        original=original,
        training=training_mean,
        test=test_mean,
        optimism=optimism,
        corrected=corrected,
        lower=lower,
        upper=upper,
        counts=np.full(len(_METRICS), len(training), dtype=np.int64),
        failed_replicates=tuple(failures),
    )
