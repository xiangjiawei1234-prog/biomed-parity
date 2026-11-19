"""Local variance-stabilizing transformation with a supplied dispersion trend."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PPoly
from scipy.linalg import solve_banded


Array = NDArray[np.float64]
_GRID_SIZE = 1000


def _finite_real(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain finite real values")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    return array


@dataclass(frozen=True)
class DispersionGrid:
    """A fitted dispersion trend sampled at increasing normalized means.

    Interpolation is linear in the normalized mean. Sample the fitted trend
    at local_vst_grid(...) to avoid interpolation on the integration grid.
    """

    means: ArrayLike
    dispersions: ArrayLike

    def __post_init__(self):
        means = _finite_real(self.means, "means")
        dispersions = _finite_real(self.dispersions, "dispersions")
        if (
            means.ndim != 1 or len(means) < 2 or dispersions.shape != means.shape
            or np.any(means < 0) or np.any(np.diff(means) <= 0)
        ):
            raise ValueError("dispersion grid requires two or more strictly increasing nonnegative means")
        if np.any(dispersions < 0):
            raise ValueError("dispersions must be nonnegative")
        object.__setattr__(self, "means", means.copy())
        object.__setattr__(self, "dispersions", dispersions.copy())

    def __call__(self, means: ArrayLike) -> Array:
        query = _finite_real(means, "query means")
        lo, hi = self.means[0], self.means[-1]
        rtol = 16 * np.finfo(float).eps
        if np.any(query < lo * (1 - rtol)) or np.any(query > hi * (1 + rtol)):
            raise ValueError("dispersion grid must cover the entire integration grid")
        return np.interp(query, self.means, self.dispersions)


def _counts_matrix(counts):
    counts = _finite_real(counts, "counts")
    if counts.ndim != 2 or not counts.size or np.any(counts < 0):
        raise ValueError("counts must be a nonempty nonnegative feature-by-sample matrix")
    return counts


def _grid_from_maximum(maximum):
    if maximum <= 0:
        raise ValueError("the matrix must contain a positive normalized count")
    with np.errstate(over="ignore"):
        grid = np.sinh(np.linspace(0, np.arcsinh(maximum), _GRID_SIZE)[1:])
    grid[-1] = maximum
    if np.any(grid <= 0) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("the count range does not support the local VST integration grid")
    return grid


def local_vst_grid(normalized_counts: ArrayLike) -> Array:
    """Return the dispersion evaluation means for a normalized count matrix."""
    return _grid_from_maximum(_counts_matrix(normalized_counts).max())


def _normalize(counts, size_factors, normalization_factors, normalized):
    size = None
    matrix = None
    if size_factors is not None:
        size = _finite_real(size_factors, "size_factors")
        if size.shape != (counts.shape[1],) or np.any(size <= 0):
            raise ValueError("size_factors must have one positive value per sample column")
    if normalization_factors is not None:
        matrix = _finite_real(normalization_factors, "normalization_factors")
        if matrix.shape != counts.shape or np.any(matrix <= 0):
            raise ValueError("normalization_factors must match counts and have positive values")
    with np.errstate(divide="ignore", over="ignore"):
        if not normalized:
            scale = matrix if matrix is not None else (size if size is not None else 1.0)
            counts = counts / scale
        if size is None:
            size = np.ones(counts.shape[1]) if matrix is None else np.exp(np.log(matrix).mean(axis=0))
        poisson_scale = float(np.mean(1 / size))
    if not np.isfinite(poisson_scale) or poisson_scale <= 0 or not np.isfinite(counts).all():
        raise ValueError("normalization resulted in nonfinite counts or Poisson scaling")
    return counts, poisson_scale


def _fmm_spline(x, y):
    """Cubic interpolation with FMM endpoint third derivatives."""
    widths = np.diff(x)
    slopes = np.diff(y) / widths
    second_divided = np.diff(slopes) / (x[2:] - x[:-2])
    third_left = (second_divided[1] - second_divided[0]) / (x[3] - x[0])
    third_right = (second_divided[-1] - second_divided[-2]) / (x[-1] - x[-4])
    rhs = np.r_[widths[0] * third_left, np.diff(slopes), widths[-1] * third_right]

    # Unknowns are the second derivatives at the knots divided by six.
    band = np.zeros((3, len(x)))
    band[0, 2:] = widths[1:]
    band[0, 1] = 1
    band[1] = np.r_[-1, 2 * (widths[:-1] + widths[1:]), 1]
    band[2, :-2] = widths[:-1]
    band[2, -2] = -1
    curvature = solve_banded((1, 1), band, rhs)
    coefficients = np.vstack((
        np.diff(curvature) / widths,
        3 * curvature[:-1],
        slopes - widths * (2 * curvature[:-1] + curvature[1:]),
        y[:-1],
    ))
    return PPoly(coefficients, x)


def local_vst(
    counts: ArrayLike,
    dispersion: Callable[[Array], ArrayLike] | DispersionGrid,
    *,
    size_factors: ArrayLike | None = None,
    normalization_factors: ArrayLike | None = None,
    normalized: bool = False,
) -> Array:
    """Apply the DESeq2 local VST to a feature-by-sample expression matrix.

    Supply an already fitted dispersion trend that accepts normalized means.
    Factors divide raw counts unless normalized is True. If both kinds of
    factors are supplied, normalize by the matrix and use sample size factors
    in the Poisson contribution. With only a matrix, use its column geometric
    means in the Poisson contribution. Default to unit sample size factors.
    """
    counts = _counts_matrix(counts)
    if not isinstance(normalized, (bool, np.bool_)):
        raise ValueError("normalized must be boolean")
    means, poisson_scale = _normalize(counts, size_factors, normalization_factors, normalized)
    grid = _grid_from_maximum(means.max())
    with np.errstate(over="ignore", invalid="ignore"):
        h1, h2 = np.quantile(means.mean(axis=1), (0.95, 0.999))
    if not np.isfinite(h2) or not 0 < h1 < h2:
        raise ValueError("the 95% and 99.9% feature-mean quantiles must be positive and distinct")

    if not callable(dispersion):
        raise ValueError("dispersion must be a callable trend or a DispersionGrid")
    values = _finite_real(dispersion(grid), "fitted dispersions")
    if (values.ndim and values.shape != grid.shape) or np.any(values < 0):
        raise ValueError("dispersion must return one nonnegative value per grid mean or a scalar")
    with np.errstate(over="ignore", divide="ignore"):
        variance_per_mean = poisson_scale + grid * values
        integrand = np.reciprocal(np.sqrt(variance_per_mean) * np.sqrt(grid))
    if not np.all(np.isfinite(integrand) & (integrand > 0)):
        raise ValueError("the fitted variance is outside the supported numerical range")
    widths = np.diff(grid)
    integrated = np.cumsum(widths * (integrand[:-1] + integrand[1:]) / 2)
    midpoints = grid[:-1] + widths / 2
    if not np.all(np.isfinite(integrated)):
        raise ValueError("the local VST integral exceeds the supported numerical range")
    spline = _fmm_spline(np.arcsinh(midpoints), integrated)
    first, last = spline(np.arcsinh([h1, h2]))
    if not last > first:
        raise ValueError("the variance spline does not give increasing anchor values")
    scale = (np.log2(h2) - np.log2(h1)) / (last - first)
    return np.log2(h1) + scale * (spline(np.arcsinh(means)) - first)
