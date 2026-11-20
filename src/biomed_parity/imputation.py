"""Pooled inference for repeated complete-data estimates and model tests."""

import warnings
from dataclasses import dataclass
from math import inf, isfinite, sqrt
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2
from scipy.stats import f as fisher_f
from scipy.stats import t as student_t


@dataclass(frozen=True)
class PooledScalar:
    m: int
    qhat: tuple[float, ...]
    u: tuple[float, ...]
    qbar: float
    ubar: float
    b: float
    t: float
    r: float
    df: float
    fmi: float
    standard_error: float
    statistic: float
    pvalue: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PooledParameters:
    estimate: NDArray[np.float64]
    within_covariance: NDArray[np.float64]
    between_covariance: NDArray[np.float64]
    total_covariance: NDArray[np.float64]
    marginal: tuple[PooledScalar, ...]


@dataclass(frozen=True)
class PooledTest:
    statistic: float
    df1: int
    df2: float
    pvalue: float
    relative_increase: float


def _real_vector(values: ArrayLike, name: str) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim != 1 or data.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a one-dimensional vector of finite real numbers")
    data = data.astype(float)
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must be a one-dimensional vector of finite real numbers")
    return data


def pool_scalar(
    q: ArrayLike,
    u: ArrayLike,
    *,
    n: float = inf,
    k: int = 1,
    rule: str = "rubin1987",
    confidence_level: float = 0.95,
    null: float = 0.0,
) -> PooledScalar:
    """Pool row-aligned scalar estimates and variances across at least two fits.

    Use Rubin's rule for multiple imputation and Reiter's rule for partially
    synthetic data from a fully observed dataset. Bounds and tests are two-sided.
    """
    estimates = _real_vector(q, "q")
    variances = _real_vector(u, "u")
    m = len(estimates)
    if len(variances) != m or m < 2:
        raise ValueError("q and u must have the same length of at least two")
    if np.any(variances < 0):
        raise ValueError("u must contain nonnegative variances")
    if (
        not isinstance(n, Real) or not isinstance(k, (int, np.integer))
        or isinstance(k, bool) or not 1 <= k < n
    ):
        raise ValueError("n must exceed the positive integer number of parameters k")
    if (
        not isinstance(confidence_level, Real) or not 0 < confidence_level < 1
        or not isinstance(null, Real) or not isfinite(null)
    ):
        raise ValueError("confidence_level must be between 0 and 1 and null must be finite")
    if rule not in ("rubin1987", "reiter2003"):
        raise ValueError("rule must be 'rubin1987' or 'reiter2003'")

    deltas = estimates - estimates[0]
    qbar = float(estimates[0] + deltas.mean())
    ubar = float(variances.mean())
    between = float(deltas.var(ddof=1))
    added = between * (1 + 1 / m)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        relative = float(np.divide(added, ubar))
        if rule == "rubin1987":
            total = ubar + added
            missing = np.divide(added, total)
            old_df = np.divide(m - 1, missing**2)
            df = old_df
            if isfinite(n):
                complete_df = n - k
                observed_df = (complete_df + 1) / (complete_df + 3) * complete_df * (1 - missing)
                df = np.divide(1, 1 / old_df + np.divide(1, observed_df))
            fmi = float(np.divide(relative + 2 / (df + 3), relative + 1))
        else:
            synthetic_component = between / m
            total = ubar + synthetic_component
            df = (m - 1) * np.divide(total, synthetic_component) ** 2
            fmi = float("nan")

        standard_error = sqrt(total)
        statistic = float(np.divide(qbar - null, standard_error))
        pvalue = float(2 * student_t.sf(abs(statistic), df))
        margin = float(student_t.isf((1 - confidence_level) / 2, df)) * standard_error

    return PooledScalar(
        m, tuple(map(float, estimates)), tuple(map(float, variances)),
        qbar, ubar, between, total, relative, float(df), fmi,
        standard_error, statistic, pvalue, qbar - margin, qbar + margin,
    )


def _coefficient_inputs(estimates, covariances):
    q = np.asarray(estimates)
    u = np.asarray(covariances)
    for array in (q, u):
        if array.dtype.kind not in "buif" or not np.isfinite(array).all():
            raise ValueError("estimates and covariances must be finite real arrays")
    if q.ndim != 2 or len(q) < 2 or not q.shape[1]:
        raise ValueError("estimates must have one row per fit and one column per parameter")
    if u.shape != (len(q), q.shape[1], q.shape[1]):
        raise ValueError("covariances must have one square parameter covariance per fit")
    q, u = q.astype(float), u.astype(float)
    scale = np.max(abs(u), axis=(1, 2))
    if np.any(np.max(abs(u - u.transpose(0, 2, 1)), axis=(1, 2)) > 1e-12 * scale):
        raise ValueError("each covariance must be symmetric")
    u = (u + u.transpose(0, 2, 1)) / 2
    if np.any(np.linalg.eigvalsh(u)[:, 0] < -1e-12 * scale):
        raise ValueError("each covariance must be positive semidefinite")
    if np.any(np.diagonal(u, axis1=1, axis2=2) < 0):
        raise ValueError("covariance diagonal entries must be nonnegative")
    return q, u


def _complete_df(degrees):
    if not isinstance(degrees, Real) or not 0 < degrees:
        raise ValueError("complete_df must be positive or infinity")
    return float(degrees)


def _moments(estimates, covariances):
    delta = estimates - estimates[0]
    mean_delta = delta.mean(axis=0)
    estimate = estimates[0] + mean_delta
    centered = delta - mean_delta
    between = centered.T @ centered / (len(estimates) - 1)
    return estimate, covariances.mean(axis=0), between


def pool_parameters(
    estimates: ArrayLike,
    covariances: ArrayLike,
    *,
    complete_df: float = inf,
    confidence_level: float = 0.95,
) -> PooledParameters:
    """Pool aligned vector estimates, full covariances and marginal inference.

    Inputs have shape fits × parameters and fits × parameters × parameters.
    The within- and between-imputation covariances follow Rubin's rules.
    """
    q, u = _coefficient_inputs(estimates, covariances)
    degrees = _complete_df(complete_df)
    point, within, between = _moments(q, u)
    marginal = tuple(
        pool_scalar(
            q[:, j], u[:, j, j], n=degrees + 1, confidence_level=confidence_level,
        )
        for j in range(q.shape[1])
    )
    return PooledParameters(point, within, between, within + (1 + 1 / len(q)) * between, marginal)


def _d1_degrees(relative, constraints, fits, complete_df):
    t = constraints * (fits - 1)
    if not isfinite(complete_df):
        if relative == 0:
            return inf
        if t <= 4:
            return t * (1 + 1 / constraints) * (1 + 1 / relative)**2 / 2
        return 4 + (t - 4) * (1 + (1 - 2 / t) / relative)**2

    effective = complete_df * (1 - 2 / (complete_df + 3))
    if relative == 0:
        return effective
    # Reiter's finite-complete-data-df moment adjustment.
    if t in (2, 4):
        return float("nan")
    a = relative * t / (t - 2)
    term = 1 + a
    c1, c2 = effective - 2 * term, effective - 4 * term
    if not c1 or not c2 or not term:
        return float("nan")
    scaled = a**2 / (t - 4)
    correction = (
        (1 + scaled * (c1 / term**2 + 4 / term + 4 / c1)) / c2
        + scaled * (8 * c1 / term + 8) / c2**2
        + scaled * 16 * c1 / c2**3
    )
    return 4 + 1 / correction if correction else inf


def _pooled_test(statistic, numerator_df, denominator_df, relative):
    if denominator_df == inf:
        pvalue = chi2.sf(statistic * numerator_df, df=numerator_df)
    else:
        pvalue = fisher_f.sf(statistic, numerator_df, denominator_df)
    return PooledTest(float(statistic), int(numerator_df), float(denominator_df),
                      float(pvalue), float(relative))


def pool_d1(
    estimates: ArrayLike,
    covariances: ArrayLike,
    *,
    contrast: ArrayLike | None = None,
    null: ArrayLike = 0,
    complete_df: float = inf,
) -> PooledTest:
    """Test a joint linear null after multiple imputation using D1.

    Contrast rows define hypotheses on the parameter vector. The null value
    may be scalar or one value per contrast row. The default tests all
    parameters jointly against zero.
    """
    q, u = _coefficient_inputs(estimates, covariances)
    degrees = _complete_df(complete_df)
    matrix = np.eye(q.shape[1]) if contrast is None else np.atleast_2d(np.asarray(contrast))
    if (
        matrix.ndim != 2 or matrix.shape[1] != q.shape[1] or not len(matrix)
        or matrix.dtype.kind not in "buif" or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != len(matrix)
    ):
        raise ValueError("contrast must have linearly independent parameter-aligned rows")
    target = np.asarray(null)
    if (
        target.shape not in ((), (len(matrix),)) or target.dtype.kind not in "buif"
        or not np.isfinite(target).all()
    ):
        raise ValueError("null must be finite and match the number of contrast rows")
    # A fixed null shift changes only the mean, leaving the covariance components intact.
    estimate, within, between = _moments(q @ matrix.T, matrix @ u @ matrix.T)
    difference = estimate - target
    try:
        factor = cho_factor(within)
    except np.linalg.LinAlgError as error:
        raise ValueError("the average within-imputation contrast covariance is singular") from error
    k = len(matrix)
    relative = max(0.0, (1 + 1 / len(q)) * np.trace(cho_solve(factor, between)) / k)
    statistic = difference @ cho_solve(factor, difference) / (k * (1 + relative))
    return _pooled_test(statistic, k, _d1_degrees(relative, k, len(q), degrees), relative)


def pool_d2(statistics: ArrayLike, df: int) -> PooledTest:
    """Combine repeated Wald or likelihood-ratio chi-square tests using D2."""
    values = _real_vector(statistics, "statistics")
    if len(values) < 2 or np.any(values < 0):
        raise ValueError("statistics must contain two or more nonnegative test values")
    if not isinstance(df, Integral) or isinstance(df, bool) or df < 1:
        raise ValueError("df must be a positive integer")
    m = len(values)
    roots = np.sqrt(values)
    relative = float((1 + 1 / m) * (roots - roots[0]).var(ddof=1))
    statistic = (values.mean() / df - (m + 1) / (m - 1) * relative) / (1 + relative)
    denominator_df = inf if relative == 0 else df**(-3 / m) * (m - 1) * (1 + 1 / relative)**2
    return _pooled_test(statistic, df, denominator_df, relative)


def _nested_loglikelihoods(full_loglik, reduced_loglik, df):
    full = _real_vector(full_loglik, "full_loglik")
    reduced = _real_vector(reduced_loglik, "reduced_loglik")
    if full.size < 2 or full.shape != reduced.shape:
        raise ValueError("full and reduced log likelihoods must align across at least two imputations")
    if not isinstance(df, Integral) or isinstance(df, (bool, np.bool_)) or df < 1:
        raise ValueError("df must be the positive integer difference in model parameter counts")
    return full, reduced


def pool_d3(
    full_loglik: ArrayLike,
    reduced_loglik: ArrayLike,
    restricted_full_loglik: ArrayLike,
    restricted_reduced_loglik: ArrayLike,
    df: int,
    *,
    ariv: str = "default",
) -> PooledTest:
    """Pool nested log likelihoods using the Meng–Rubin D3 test.

    The first pair contains per-imputation maximum log likelihoods. The
    restricted pair evaluates each model on the same imputations under its
    own across-imputation pooled parameters. The caller selects how to
    restrict nuisance and regression parameters when calculating those inputs.
    Use ``ariv="positive"`` to truncate negative relative variance increases.
    """
    full, reduced = _nested_loglikelihoods(full_loglik, reduced_loglik, df)
    restricted_full = _real_vector(restricted_full_loglik, "restricted_full_loglik")
    restricted_reduced = _real_vector(restricted_reduced_loglik, "restricted_reduced_loglik")
    if restricted_full.shape != full.shape or restricted_reduced.shape != reduced.shape:
        raise ValueError("restricted log likelihoods must align with the fitted log likelihoods")
    if ariv not in ("default", "positive"):
        raise ValueError("ariv must be 'default' or 'positive' for D3")

    m = full.size
    fitted = 2 * (full - reduced).mean()
    restricted = 2 * (restricted_full - restricted_reduced).mean()
    relative = (m + 1) / (df * (m - 1)) * (fitted - restricted)
    if ariv == "positive":
        relative = np.maximum(relative, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        statistic = restricted / (df * (1 + relative))
        denominator_df = _d1_degrees(relative, df, m, inf)
    if np.isfinite(statistic) and statistic < 0:
        warnings.warn("D3 statistic is negative; inspect the restricted model likelihoods", RuntimeWarning,
                      stacklevel=2)
    return _pooled_test(statistic, df, denominator_df, relative)


def pool_d4(
    full_loglik: ArrayLike,
    reduced_loglik: ArrayLike,
    stacked_full_loglik: float,
    stacked_reduced_loglik: float,
    df: int,
    *,
    ariv: str = "default",
    full_model_df: int | None = None,
) -> PooledTest:
    """Pool nested log likelihoods using the Chan–Meng D4 test in mitml.

    Stack all completed datasets row-wise and fit each model once, with
    parameters shared across imputations. Supply the resulting unscaled
    stacked log likelihoods along with the per-imputation fits. ``ariv``
    selects the default, nonnegative (``"positive"``), or ``"robust"``
    variance increase. The robust method requires the full model's total
    parameter count, including estimated nuisance parameters.
    """
    full, reduced = _nested_loglikelihoods(full_loglik, reduced_loglik, df)
    if ariv not in ("default", "positive", "robust"):
        raise ValueError("ariv must be 'default', 'positive' or 'robust' for D4")
    for value in (stacked_full_loglik, stacked_reduced_loglik):
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not isfinite(value):
            raise ValueError("stacked log likelihoods must be finite numbers")

    m = full.size
    # mitml removes differences considered equal by R's all.equal() before pooling.
    tolerance = sqrt(np.finfo(float).eps)
    stacked_difference = (stacked_full_loglik - stacked_reduced_loglik) / m
    stacked = 0.0 if abs(stacked_difference) <= tolerance else 2 * stacked_difference
    if ariv == "robust":
        if (
            not isinstance(full_model_df, Integral) or isinstance(full_model_df, (bool, np.bool_))
            or full_model_df < df
        ):
            raise ValueError("full_model_df must be an integer at least as large as df for robust D4")
        constraints = full_model_df
        loss = 2 * (full - stacked_full_loglik / m).mean()
    else:
        constraints = df
        difference = full - reduced
        scale = np.where(abs(full) > tolerance, abs(full), 1.0)
        fitted = 2 * np.where(abs(difference) <= tolerance * scale, 0.0, difference).mean()
        loss = fitted - stacked

    adjusted_df = constraints * (m - 1)
    relative = (m + 1) / adjusted_df * loss
    if ariv == "positive":
        relative = np.maximum(relative, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator_df = inf if relative == 0 else adjusted_df * (1 + 1 / relative)**2
        statistic = stacked / (df * (1 + relative))
    return _pooled_test(statistic, df, denominator_df, relative)
