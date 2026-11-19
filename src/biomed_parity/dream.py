"""Precision-weighted Gaussian REML with one random intercept."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import t

from ._survival_inputs import _encode_labels, _numeric_vector


@dataclass(frozen=True)
class RandomInterceptResult:
    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    residual_variance: float
    random_intercept_variance: float
    estimate: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    df: NDArray[np.float64]
    statistic: NDArray[np.float64]
    pvalue: NDArray[np.float64]


def _numeric_matrix(matrix, name):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not matrix.size or matrix.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty numeric matrix")
    matrix = matrix.astype(float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _residual_spectrum(x, y, z):
    q, _ = np.linalg.qr(x, mode="reduced")
    residual_z = z - q @ (q.T @ z)
    residual_y = y - q @ (q.T @ y)
    total = residual_y @ residual_y
    if total <= np.finfo(float).eps**2 * (y @ y):
        raise ValueError("a finite REML fit requires nonzero residual variation")

    crossproduct = np.einsum("ij,ik->jk", residual_z, residual_z)
    eigenvalues, vectors = np.linalg.eigh(crossproduct)
    largest = eigenvalues[-1]
    null_tolerance = np.finfo(float).eps * max(z.shape) * np.max(np.sum(z**2, axis=0))
    active = eigenvalues > null_tolerance
    if not np.any(active):
        raise ValueError("the random intercept has no variation outside the fixed design")
    eigenvalues = eigenvalues[active]
    power = (vectors[:, active].T @ (residual_z.T @ residual_y))**2 / eigenvalues
    residual_df = x.shape[0] - x.shape[1]
    nullity = residual_df - len(eigenvalues)
    if nullity == 0 and np.allclose(eigenvalues, largest, rtol=1e-12, atol=0):
        raise ValueError("residual variance and random intercept variance are unidentifiable")
    power = np.r_[max(0, total - power.sum()), power]
    multiplicity = np.r_[nullity, np.ones(len(eigenvalues))]
    return np.r_[0, eigenvalues], power, multiplicity


def _variance_components(eigenvalues, power, multiplicity):
    n_residual = multiplicity.sum()
    scale = eigenvalues.max()
    normalized = eigenvalues / scale

    def profile(ratio):
        inflated = 1 + ratio * normalized
        sse = np.sum(power / inflated)
        residual = sse / n_residual
        criterion = n_residual * np.log(residual) + multiplicity @ np.log(inflated)
        derivative = np.sum(normalized / inflated * (multiplicity - power / inflated / residual))
        return criterion, derivative, residual

    # Search the ratio on the information scale of the largest eigenvalue.
    # All brackets with a negative-to-positive score crossing are candidates.
    log_upper = 18 + np.log(scale / eigenvalues[eigenvalues > 0].min())
    grid = np.r_[0, np.exp(np.linspace(-14, log_upper, 80))]
    signs = np.array([profile(point)[1] for point in grid])
    candidates = [0.0]
    crossing = np.flatnonzero((signs[:-1] < 0) & (signs[1:] >= 0))
    for k in crossing:
        root = brentq(lambda r: profile(r)[1], grid[k], grid[k + 1])
        candidates.append(root)
    best = min(candidates, key=lambda r: profile(r)[0])
    loss, _, residual = profile(best)
    if profile(grid[-1])[0] < loss - 1e-8:
        raise ArithmeticError("REML optimum has vanishing residual variance")
    return float(residual), float(residual * best / scale)


def dream_random_intercept(
    y: ArrayLike,
    x: ArrayLike,
    group: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    contrasts: ArrayLike | None = None,
) -> RandomInterceptResult:
    """Fit one feature with residual variance proportional to inverse weights.

    The design matrix includes any desired intercept. Each contrast row is
    tested separately by a two-sided t test with Satterthwaite degrees of
    freedom. The boundary with zero random-intercept variance is evaluated
    as weighted least squares.
    """
    y = _numeric_vector(y, "y")
    x = _numeric_matrix(x, "x")
    n, p = x.shape
    if n != len(y) or not p < n:
        raise ValueError("x must have one row per observation and positive residual degrees of freedom")
    groups, codes = _encode_labels(group, None, n, "group")
    if not 2 <= len(groups) < n:
        raise ValueError("group must contain multiple groups and at least one repeated observation")
    if weights is None:
        w = np.ones(n)
    else:
        w = _numeric_vector(weights, "weights")
        if len(w) != n or np.any(w <= 0):
            raise ValueError("weights must be positive and match y")
    if contrasts is None:
        contrast = np.eye(p)
    else:
        contrast = _numeric_matrix(np.atleast_2d(contrasts), "contrasts")
        if contrast.shape[1] != p or np.any(np.all(contrast == 0, axis=1)):
            raise ValueError("contrast rows must be nonzero with one column per fixed coefficient")

    sw = np.sqrt(w)
    wx = sw[:, None] * x
    wy = sw * y
    if np.linalg.matrix_rank(wx) < p:
        raise ValueError("the fixed-effects design is rank deficient")
    z = np.zeros((n, len(groups)))
    z[np.arange(n), codes] = sw
    eigenvalues, power, multiplicity = _residual_spectrum(wx, wy, z)
    residual, random = _variance_components(eigenvalues, power, multiplicity)

    relative = random / residual
    adjustment = relative / (1 + relative * np.bincount(codes, weights=w))
    precision_x = (wx - z @ (adjustment[:, None] * (z.T @ wx))) / residual
    cov = np.linalg.inv(wx.T @ precision_x)
    coefficients = cov @ (precision_x.T @ wy)
    contrast_variance = np.einsum("ij,jk,ik->i", contrast, cov, contrast)
    if random == 0:
        df = np.full(len(contrast), n - p, dtype=float)
    else:
        variances = residual + random * eigenvalues
        derivatives = np.stack((np.ones_like(eigenvalues), eigenvalues), axis=1)
        hessian_term = (2 * power / variances - multiplicity) / variances**2
        hessian = derivatives.T @ (hessian_term[:, None] * derivatives)
        influence = precision_x @ cov @ contrast.T
        gradient = np.stack((np.sum(influence**2, axis=0), np.sum((z.T @ influence)**2, axis=0)))
        variance_of_variance = np.sum(gradient * np.linalg.solve(hessian, gradient), axis=0) * 2
        if np.any(variance_of_variance <= 0):
            raise np.linalg.LinAlgError("Satterthwaite variance is not positive at the REML optimum")
        df = 2 * contrast_variance**2 / variance_of_variance

    estimate = contrast @ coefficients
    standard_error = np.sqrt(contrast_variance)
    statistic = estimate / standard_error
    return RandomInterceptResult(
        coefficients=coefficients,
        covariance=cov,
        residual_variance=residual,
        random_intercept_variance=random,
        estimate=estimate,
        standard_error=standard_error,
        df=df,
        statistic=statistic,
        pvalue=2 * t.sf(np.abs(statistic), df),
    )
