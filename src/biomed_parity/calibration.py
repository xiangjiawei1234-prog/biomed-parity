"""Bootstrap internal validation of a binary logistic prediction model."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata

from ._logistic import _deviance, _fit_coefficients
from ._lowess import _lowess

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
    coefficients = _fit_coefficients(design, y)
    slopes = coefficients[1:] / spread
    return np.r_[coefficients[0] - np.sum(center * slopes), slopes]


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
        missing = np.full_like(corrected, np.nan)
        return missing.copy(), missing
    if len(values) < 10:
        top = bottom = np.std(values, axis=0, ddof=1)
    else:
        centered = values - values.mean(axis=0)
        top_mask = centered >= 0
        bottom_mask = centered <= 0
        def semideviation(side):
            count = side.sum(axis=0)
            squares = np.where(side, centered**2, 0).sum(axis=0)
            variance = np.divide(
                squares, count - 1, out=np.full_like(corrected, np.nan), where=count > 1,
            )
            return np.sqrt(variance)
        top = semideviation(top_mask)
        bottom = semideviation(bottom_mask)
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


@dataclass(frozen=True)
class LogisticCalibrationResult:
    """Pointwise estimates on the prediction grid for a binary model."""

    predy: NDArray[np.float64]
    fitted_probabilities: NDArray[np.float64]
    apparent: NDArray[np.float64]
    corrected: NDArray[np.float64]
    training_error: NDArray[np.float64]
    test_error: NDArray[np.float64]
    optimism: NDArray[np.float64]
    lower: NDArray[np.float64] | None
    upper: NDArray[np.float64] | None
    counts: NDArray[np.int64]
    failed_replicates: tuple[int, ...]


def _curve_at(probability, outcome, grid, smoother, span, iterations, delta, digits):
    if digits is not None:
        probability = np.round(probability, digits)
    if smoother is not None:
        bp, by = smoother(probability.copy(), outcome.copy())
        bp = _finite_array(bp, "smoother x")
        by = _finite_array(by, "smoother y")
        if bp.ndim != 1 or bp.shape != by.shape:
            raise ValueError("smoother must return two aligned one-dimensional arrays")
        bp, first = np.unique(bp, return_index=True)
        by = by[first]
    else:
        bp, by = _lowess(probability, outcome, span, iterations, delta)
    if len(bp) < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, bp, by, left=np.nan, right=np.nan)


def _curve_resampling(n, method, bootstrap_indices, fold_order, permutation_indices, n_resamples, seed):
    all_rows = np.arange(n)
    if method != "randomization" and permutation_indices is not None:
        raise ValueError("permutation_indices apply only to randomization")
    if method == "crossvalidation":
        if bootstrap_indices is not None:
            raise ValueError("bootstrap_indices apply to bootstrap methods")
        if not isinstance(n_resamples, Integral) or not 2 <= n_resamples <= n // 2:
            raise ValueError("crossvalidation requires between 2 and n // 2 folds in n_resamples")
        if fold_order is None:
            order = np.random.default_rng(seed).permutation(n)
        else:
            if seed is not None:
                raise ValueError("seed applies when fold_order is omitted")
            order = np.asarray(fold_order)
            if order.dtype.kind not in "iu" or not np.array_equal(np.sort(order), all_rows):
                raise ValueError("fold_order must be a permutation of all zero-based row indices")
        width = n / n_resamples
        validation = [
            order[start:min(n, round(start + width))]
            for start in (round(i * width) for i in range(n_resamples))
        ]
        training = [np.setdiff1d(all_rows, test) for test in validation]
        return zip(training, validation, np.ones(n_resamples), strict=True)
    if fold_order is not None:
        raise ValueError("fold_order applies only to crossvalidation")
    if method == "randomization":
        if bootstrap_indices is not None:
            raise ValueError("bootstrap_indices apply to bootstrap methods")
        if permutation_indices is None:
            count = 40 if n_resamples is None else n_resamples
            if not isinstance(count, Integral) or count < 1:
                raise ValueError("n_resamples must be a positive integer")
            rng = np.random.default_rng(seed)
            permutations = (rng.permutation(n) for _ in range(count))
        else:
            permutations = _resamples(n, permutation_indices, n_resamples, seed)
            if np.any(np.sort(permutations, axis=1) != all_rows):
                raise ValueError("permutation_indices must permute all zero-based row indices")
        return ((outcomes, all_rows, 1.0) for outcomes in permutations)
    bootstrap = _resamples(n, bootstrap_indices, n_resamples, seed)
    if method == "boot":
        return ((training, all_rows, 1.0) for training in bootstrap)
    omitted = np.ones(bootstrap.shape, dtype=bool)
    omitted[np.arange(len(bootstrap))[:, None], bootstrap] = False
    counts = omitted.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError(".632 requires every observation to be omitted at least once")
    weights = (omitted / counts).sum(axis=1) / n
    return zip(bootstrap, map(np.flatnonzero, omitted), weights, strict=True)


def calibrate_logistic(
    x: ArrayLike,
    y: ArrayLike,
    *,
    predy: ArrayLike | None = None,
    method: Literal["boot", ".632", "crossvalidation", "randomization"] = "boot",
    bootstrap_indices: ArrayLike | None = None,
    fold_order: ArrayLike | None = None,
    permutation_indices: ArrayLike | None = None,
    n_resamples: int | None = None,
    seed: int | np.random.Generator | None = None,
    span: float = 2 / 3,
    iterations: int = 0,
    delta: float | None = None,
    digits: int | None = None,
    smoother: Callable | None = None,
    confidence: float | None = 0.95,
) -> LogisticCalibrationResult:
    """Calibrate binary logistic probabilities on a common prediction grid.

    Supply numeric design columns without an intercept and an outcome
    encoded 0/1. The default LOWESS uses a span of 2/3 and no robust passes.
    Optionally pass a smoother that returns paired probabilities and
    fitted outcome values. Areas outside its support are not extrapolated.
    The bootstrap uses the original observations for testing, while .632
    uses held-out observations from each draw. For cross-validation,
    n_resamples gives the fold count and fold_order fixes the row order.
    Randomization permutes the training labels and tests on the original
    observations. Pass permutation_indices to supply fixed permutations.
    """
    x = _finite_array(x, "x")
    y = _finite_array(y, "y")
    if x.ndim != 2 or x.shape[1] < 1 or y.ndim != 1 or x.shape[0] != len(y) or not len(y):
        raise ValueError("x must be a nonempty design matrix with one row per y value")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("y must contain both binary outcome classes encoded as 0 and 1")
    if method not in ("boot", ".632", "crossvalidation", "randomization"):
        raise ValueError("method must be 'boot', '.632', 'crossvalidation' or 'randomization'")
    if not isinstance(span, Real) or not 0 < span <= 1:
        raise ValueError("span must be in (0, 1]")
    if not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if delta is not None and (not isinstance(delta, Real) or not np.isfinite(delta) or delta < 0):
        raise ValueError("delta must be None or finite and nonnegative")
    if digits is not None and (not isinstance(digits, Integral) or not 0 <= digits <= 15):
        raise ValueError("digits must be None or an integer between 0 and 15")
    if smoother is not None and not callable(smoother):
        raise ValueError("smoother must be None or callable")
    if confidence is not None and (not isinstance(confidence, Real) or not 0 < confidence < 1):
        raise ValueError("confidence must be None or in (0, 1)")

    original = _logistic_fit(x, y)
    predicted = expit(original[0] + x @ original[1:])
    if predy is None:
        if len(y) < 11:
            raise ValueError("supply predy when there are fewer than 11 observations")
        ordered = np.sort(predicted)
        grid = np.linspace(ordered[4], ordered[-5], 50)
    else:
        grid = np.atleast_1d(_finite_array(predy, "predy"))
        if grid.ndim != 1 or not len(grid) or np.any((grid < 0) | (grid > 1)):
            raise ValueError("predy must contain probabilities in [0, 1]")

    def curve(probabilities, outcomes):
        return _curve_at(probabilities, outcomes, grid, smoother, span, iterations, delta, digits)

    apparent = curve(predicted, y)
    samples = _curve_resampling(
        len(y), method, bootstrap_indices, fold_order, permutation_indices, n_resamples, seed,
    )
    training, testing, weights, failures = [], [], [], []
    all_rows = np.arange(len(y))
    for i, (train, test, weight) in enumerate(samples):
        train_design = all_rows if method == "randomization" else train
        try:
            coefficients = _logistic_fit(x[train_design], y[train])
        except np.linalg.LinAlgError:
            failures.append(i)
            continue
        fitted = expit(coefficients[0] + x @ coefficients[1:])
        training.append(curve(fitted[train_design], y[train]) - grid)
        testing.append(curve(fitted[test], y[test]) - grid)
        weights.append(weight)
    if not training:
        raise ValueError("no successful resampled logistic fits")

    train_stats = np.asarray(training)
    test_stats = np.asarray(testing)
    valid = np.isfinite(train_stats) & np.isfinite(test_stats)
    counts = valid.sum(axis=0)
    train_sum = np.where(valid, train_stats, 0).sum(axis=0)
    test_sum = np.asarray(weights) @ np.where(valid, test_stats, 0)
    training_error = np.divide(train_sum, counts, out=np.full_like(train_sum, np.nan), where=counts > 0)
    if method == ".632":
        test_error = test_sum
        optimism = 0.632 * (apparent - grid - test_error)
    else:
        test_error = np.divide(test_sum, counts, out=np.full_like(test_sum, np.nan), where=counts > 0)
        optimism = training_error - test_error
    corrected = apparent - optimism
    lower, upper = _intervals(train_stats, test_stats, corrected, confidence)
    return LogisticCalibrationResult(
        grid, predicted, apparent, corrected, training_error, test_error,
        optimism, lower, upper, counts, tuple(failures),
    )
