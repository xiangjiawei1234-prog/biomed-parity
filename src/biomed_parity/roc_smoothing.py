"""Binormal, density, and log-concave smoothing of binary ROC curves."""

from dataclasses import dataclass
from math import hypot, pi, sqrt
from numbers import Integral, Real
from typing import Literal
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from ._log_concave import log_concave_rates
from .roc import _binary_inputs
from .roc_bootstrap import Focus, RocDirection, _AreaSpec, _restricted_trapezoids
from .roc_thresholds import _empirical_points, _resolve_direction

SmoothingMethod = Literal["binormal", "density", "logcondens", "logcondens.smooth"]
DensityKernel = Literal[
    "gaussian", "rectangular", "triangular", "epanechnikov", "biweight", "cosine", "optcosine"
]

_KERNEL_SUPPORT = {
    "rectangular": sqrt(3),
    "triangular": sqrt(6),
    "epanechnikov": sqrt(5),
    "biweight": sqrt(7),
    "cosine": 1 / sqrt(1 / 3 - 2 / pi**2),
    "optcosine": 1 / sqrt(1 - 8 / pi**2),
}


@dataclass(frozen=True)
class SmoothROCResult:
    """Smoothed rates and area, with binormal coefficients or an estimated bandwidth."""

    specificity: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    auc: float
    direction: RocDirection
    method: SmoothingMethod
    percent: bool
    coefficients: tuple[float, float] | None
    bandwidth: float | None


def _binormal(labels, scores, direction, percent, n):
    points = _empirical_points(labels, scores, direction, percent)
    se = ndtri(points.sensitivity * (1 / points.scale))
    sp = ndtri(points.specificity * (1 / points.scale))
    interior = np.isfinite(se) & np.isfinite(sp)
    se, sp = se[interior], sp[interior]
    if se.size < 2 or np.ptp(se) == 0:
        raise ValueError("binormal ROC smoothing requires two distinct interior sensitivities")

    mean_se, mean_sp = se.mean(), sp.mean()
    centered_se, centered_sp = se - mean_se, sp - mean_sp
    slope = 0.0 if np.ptp(sp) == 0 else float(centered_se @ centered_sp / (centered_se @ centered_se))
    intercept = float(mean_sp - slope * mean_se)
    grid = np.linspace(0, 1, n)
    predicted_sp = np.empty_like(grid)
    predicted_sp[1:-1] = ndtr(intercept + slope * ndtri(grid[1:-1]))
    # Use the endpoint limits of a decreasing fit even when the slope is exactly zero.
    predicted_sp[0], predicted_sp[-1] = 1.0, 0.0
    return predicted_sp[::-1], ndtr(ndtri(grid[::-1])), (intercept, slope)


def _bandwidth(scores, bw, adjust):
    if isinstance(bw, str):
        if bw not in ("nrd0", "nrd"):
            raise ValueError("bw must be 'nrd0', 'nrd', or a positive number")
        with np.errstate(over="ignore", invalid="ignore"):
            spread = scores.std(ddof=1)
            lower, upper = np.quantile(scores, (0.25, 0.75))
            scale = min(spread, (upper - lower) / 1.34)
        if bw == "nrd0" and scale == 0:
            scale = spread or abs(scores[0]) or 1.0
        bw = (0.9 if bw == "nrd0" else 1.06) * scale * scores.size**-0.2
    if not isinstance(adjust, Real) or not np.isfinite(adjust) or adjust <= 0:
        raise ValueError("adjust must be finite and positive")
    if not isinstance(bw, Real) or not np.isfinite(bw) or bw <= 0:
        raise ValueError("bw must give a finite positive bandwidth")
    adjusted = float(bw) * float(adjust)
    if not np.isfinite(adjusted) or adjusted <= 0:
        raise ValueError("adjusted bandwidth must be finite and positive")
    return adjusted


def _kernel_values(lags, kernel):
    # Kernel normalization cancels when each class density is converted to rates.
    if kernel == "gaussian":
        with np.errstate(over="ignore"):
            return np.exp(-0.5 * lags**2)

    distance = lags / _KERNEL_SUPPORT[kernel]
    inside = distance < 1
    u = distance[inside]
    values = np.zeros_like(lags)
    if kernel == "rectangular":
        values[inside] = 1
    elif kernel == "triangular":
        values[inside] = 1 - u
    elif kernel == "epanechnikov":
        values[inside] = 1 - u**2
    elif kernel == "biweight":
        values[inside] = (1 - u**2)**2
    elif kernel == "cosine":
        values[inside] = 1 + np.cos(pi * u)
    else:
        values[inside] = np.cos((pi / 2) * u)
    return values


def _binned_density(scores, spectrum, internal_grid, output_grid):
    size = internal_grid.size
    step = (internal_grid[-1] - internal_grid[0]) / (size - 1)
    location = (scores - internal_grid[0]) / step
    index = location.astype(np.intp)
    fraction = location - index
    weights = np.bincount(index, weights=1 - fraction, minlength=size)
    weights += np.bincount(index + 1, weights=fraction, minlength=size + 1)[:size]
    convolution = np.fft.irfft(np.fft.rfft(weights / scores.size, n=2 * size) * spectrum)
    return np.interp(output_grid, internal_grid, np.maximum(convolution[:size], 0))


def _kernel_densities(labels, scores, n, bw, adjust, cut, kernel, control_density, case_density):
    if not isinstance(kernel, str) or (kernel != "gaussian" and kernel not in _KERNEL_SUPPORT):
        raise ValueError("kernel must be one of the seven supported density kernels")
    if not isinstance(cut, Real) or not np.isfinite(cut) or cut < 0:
        raise ValueError("cut must be finite and nonnegative")
    controls, cases = scores[labels == 0], scores[labels == 1]
    # Pool classes for a common bandwidth, using the first control for nrd0's fallback.
    bandwidth = _bandwidth(np.concatenate((controls, cases)), bw, adjust)
    start = float(scores.min()) - float(cut) * bandwidth
    stop = float(scores.max()) + float(cut) * bandwidth
    lower, upper = start - 4 * bandwidth, stop + 4 * bandwidth
    span = upper - lower
    if not np.isfinite(span) or span <= 0 or start > stop:
        raise ValueError("density smoothing requires a finite, resolvable score grid")

    # R density evaluates an FFT convolution on a larger power-of-two grid,
    # then interpolates both densities back onto the same output grid.
    size = max(512, 1 << (n - 1).bit_length())
    internal_grid = np.linspace(lower, upper, size)
    output_grid = np.linspace(start, stop, n)
    if np.any(np.diff(internal_grid) <= 0):
        raise ValueError("density smoothing requires a finite, resolvable score grid")
    with np.errstate(over="ignore"):
        lags = np.arange(size + 1) * (span / (size - 1)) / bandwidth
    positive_kernel = _kernel_values(lags, kernel)
    spectrum = np.fft.rfft(np.r_[positive_kernel, positive_kernel[-2:0:-1]])
    if control_density is None:
        control_density = _binned_density(controls, spectrum, internal_grid, output_grid)
    if case_density is None:
        case_density = _binned_density(cases, spectrum, internal_grid, output_grid)
    return control_density, case_density, bandwidth


def _tabulated_density(values, name):
    if values is None:
        return None
    values = np.asarray(values)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector of finite nonnegative densities")
    return values.astype(np.result_type(values.dtype, np.float64), copy=False)


def _density_cumulative(density, *, reverse):
    peak = density.max()
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("density grid must have positive finite mass for both classes")
    # Rescale before accumulating so large density values cannot overflow.
    ordered = density[::-1] if reverse else density
    cumulative = np.cumsum(ordered / peak, dtype=float)
    return cumulative / cumulative[-1]


def _density_rates(control_density, case_density, direction):
    if control_density.size != case_density.size:
        raise ValueError("density_controls and density_cases must have the same length")
    # The threshold's own grid bin counts toward sensitivity in either direction.
    negatives = _density_cumulative(control_density, reverse=direction == ">")
    positives = _density_cumulative(case_density, reverse=direction == "<")
    if direction == "<":
        return np.r_[0, negatives[:-1]], positives[::-1]
    return np.r_[negatives[:-1][::-1], 0], positives


def _auc(specificity, sensitivity, coefficients, area, allow_below_chance):
    if area.bounds is None and coefficients is not None:
        intercept, slope = coefficients
        return float(ndtr(intercept / hypot(1, slope))) * area.scale

    axis, height = specificity, sensitivity
    if area.focus == "sensitivity" and area.bounds is not None:
        axis, height = sensitivity[::-1], specificity[::-1]
    low, high = area.bounds or (0.0, 1.0)
    value = _restricted_trapezoids(axis, height, low, high)
    if area.bounds is not None and area.standardize:
        width = high - low
        midpoint = (low + high) / 2
        maximal_gain = width * midpoint
        if maximal_gain == 0:
            return float("nan")
        gained = value - width * (1 - midpoint)
        if abs(gained) <= 8 * np.finfo(float).eps * width:
            gained = 0.0
        if gained < 0 and not allow_below_chance:
            warn("partial AUC correction is undefined below the chance diagonal", RuntimeWarning, stacklevel=3)
            return float("nan")
        value = (1 + gained / maximal_gain) / 2
    return value * area.scale


def smooth_roc(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    method: SmoothingMethod = "binormal",
    direction: RocDirection = "<",
    n: int = 512,
    bw: str | float = "nrd0",
    kernel: DensityKernel = "gaussian",
    adjust: float = 1.0,
    cut: float = 3.0,
    density_controls: ArrayLike | None = None,
    density_cases: ArrayLike | None = None,
    partial_auc: tuple[float, float] | None = None,
    focus: Focus = "specificity",
    standardize: bool = False,
    allow_below_chance: bool = False,
    percent: bool = False,
) -> SmoothROCResult:
    """Smooth a binary ROC and compute its full or requested partial AUC.

    Labels use 0 for controls and 1 for cases. Rates are ordered by
    increasing specificity and include two anchors. With method="density",
    density_controls and density_cases can supply values on the same evenly
    spaced grid, ordered by increasing score. If either is omitted, estimate
    it on the common n-point kernel grid. With both supplied, their length
    sets the number of fitted points and bandwidth is None. Percent mode
    uses the 0–100 convention.

    The log-concave methods use pROC's high-score curve convention, even
    when the original ROC direction resolves to ">".
    """
    if method not in ("binormal", "density", "logcondens", "logcondens.smooth"):
        raise ValueError("method must be 'binormal', 'density', 'logcondens', or 'logcondens.smooth'")
    if method != "density" and (density_controls is not None or density_cases is not None):
        raise ValueError("density_controls and density_cases require method='density'")
    uses_kernel = method == "density" and (density_controls is None or density_cases is None)
    if method != "density" or uses_kernel:
        minimum_n = 1 if method == "density" else 2
        if not isinstance(n, Integral) or isinstance(n, bool) or n < minimum_n:
            raise ValueError(f"n must be an integer of at least {minimum_n} for {method} smoothing")
    labels, scores = _binary_inputs(labels, scores)
    scores = scores.astype(float, copy=False)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("ROC requires at least one case and one control")
    direction = _resolve_direction(labels == 1, scores, direction)
    area = _AreaSpec.create(partial_auc, focus, standardize, percent)
    coefficients, bandwidth = None, None
    if method == "binormal":
        sp, se, coefficients = _binormal(labels, scores, direction, percent, int(n))
    elif method == "density":
        controls = _tabulated_density(density_controls, "density_controls")
        cases = _tabulated_density(density_cases, "density_cases")
        if uses_kernel:
            controls, cases, bandwidth = _kernel_densities(
                labels, scores, int(n), bw, adjust, cut, kernel, controls, cases,
            )
        sp, se = _density_rates(controls, cases, direction)
    else:
        sp, se = log_concave_rates(scores[labels == 0], scores[labels == 1], int(n),
            smooth=method == "logcondens.smooth")
    if np.any(np.diff(sp) < 0):
        sp, se = sp[::-1], se[::-1]
    sp, se = np.r_[0.0, sp, 1.0], np.r_[1.0, se, 0.0]
    value = _auc(sp, se, coefficients, area, allow_below_chance)
    return SmoothROCResult(sp * area.scale, se * area.scale, value, direction, method, percent, coefficients, bandwidth)
