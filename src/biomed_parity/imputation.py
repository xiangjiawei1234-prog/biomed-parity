"""Scalar inference for repeated complete-data estimates."""

from dataclasses import dataclass
from math import inf, isfinite, sqrt
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
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

    qbar = float(estimates.mean())
    ubar = float(variances.mean())
    between = float(estimates.var(ddof=1))
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
