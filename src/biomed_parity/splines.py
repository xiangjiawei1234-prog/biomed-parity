"""Harrell-style restricted cubic spline bases and reusable transforms."""

from dataclasses import dataclass
from math import prod
from numbers import Integral, Real
from typing import Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SplineKnotsWarning(UserWarning):
    """The observed support required a change to the default knot rule."""


def _vector(x: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.atleast_1d(np.asarray(x))
    if values.ndim != 1:
        raise ValueError(f"{name} must be a real numeric vector")
    if values.dtype.kind not in "biuf" and (
        values.dtype.kind != "O" or not all(value is None or isinstance(value, Real) for value in values)
    ):
        raise ValueError(f"{name} must be a real numeric vector")
    values = values.astype(float)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinite values")
    return values


def _check_knots(values):
    knots = np.unique(values[~np.isnan(values)])
    if len(knots) < 3:
        raise ValueError("fewer than three distinct knots")
    if not np.isfinite(float(knots[-1]) - float(knots[0])):
        raise ValueError("knot spacing is outside the finite numeric range")
    return knots


def _quantile_knots(sample, count, outer, first, last):
    if len(sample) == 0:
        raise ValueError("the tie exclusions leave no observations to estimate interior knots")
    if count == 1:
        return np.asarray([np.median(sample)])
    start = 0.5 if count == 2 else outer
    knots = np.quantile(sample, np.linspace(start, 1 - outer, count))
    if len(np.unique(knots)) < min(count, 3):
        knots = np.quantile(sample, np.linspace(outer, 1 - outer, 2 * count))
        if first is not None and len(np.unique(knots)) < 3:
            middle = (first + last) / 2 if last is not None else np.median(sample)
            upper = last if last is not None else np.quantile(sample, 1 - outer)
            knots = np.sort(np.asarray([first, middle, upper]))
        unique = len(np.unique(knots))
        if unique < 3:
            raise ValueError("fewer than three distinct knots after the fallback quantiles")
        warnings.warn(
            f"Could not obtain {count} interior knots with the default quantiles. "
            f"The fallback found {unique} knots.",
            SplineKnotsWarning, stacklevel=3,
        )
    if len(sample) < 100:
        if first is None:
            knots[0] = sample[4] if len(sample) >= 5 else np.nan
        if last is None:
            knots[count - 1] = sample[-5] if len(sample) >= 5 else np.nan
    return knots


def restricted_cubic_spline_knots(
    x: ArrayLike, *, n_knots: int = 5, tie_fraction: float = 0.05,
) -> NDArray[np.float64]:
    """Estimate knots by the Hmisc quantile and extreme-tie rules."""
    if not isinstance(n_knots, Integral) or n_knots < 3:
        raise ValueError("n_knots must be an integer of at least three")
    if not isinstance(tie_fraction, Real) or not 0 <= tie_fraction <= 1:
        raise ValueError("tie_fraction must be in [0, 1]")
    values = _vector(x, "x")
    sample = np.sort(values[~np.isnan(values)])
    if len(sample) < 6:
        raise ValueError("at least six nonmissing observations are required to estimate knots")
    unique, counts = np.unique(sample, return_counts=True)
    if len(unique) - 2 <= n_knots:
        warnings.warn(
            f"{n_knots} knots requested with {len(unique)} unique values. "
            f"Using the {max(len(unique) - 2, 0)} interior values.",
            SplineKnotsWarning, stacklevel=2,
        )
        return _check_knots(unique[1:-1])

    first = last = None
    if 0 < tie_fraction < 1 and counts[1:-1].max() / len(sample) < tie_fraction:
        if counts[0] / len(sample) >= tie_fraction:
            first = unique[1]
        if counts[-1] / len(sample) >= tie_fraction:
            last = unique[-2]
        if first is not None:
            sample = sample[sample > first]
        if last is not None:
            sample = sample[sample < last]
    outer = 0.1 if n_knots == 3 else (0.025 if n_knots > 6 else 0.05)
    effective = n_knots - (first is not None) - (last is not None)
    interior = _quantile_knots(sample, effective, outer, first, last)
    anchors = np.array([knot for knot in (first, last) if knot is not None])
    return _check_knots(np.r_[interior, anchors])


def _nonlinear_basis(x, knots, scale, power):
    basis = np.empty((len(x), len(knots) - 2))
    tail = x >= knots[-1]
    interior = x[~tail, None]
    a = (knots[-1] - knots[:-2]) / scale
    b = (knots[-1] - knots[-2]) / scale
    # The terms at the final knot vanish on the interior side.
    basis[~tail] = (
        np.maximum((interior - knots[:-2]) / scale, 0)**power
        - (a / b) * np.maximum((interior - knots[-2]) / scale, 0)**power
    )
    # Expand at the final knot to evaluate the exact polynomial tail without
    # subtracting nearly equal cubes at large prediction values.
    distance = (x[tail, None] - knots[-1]) / scale
    product = a * ((knots[-2] - knots[:-2]) / scale)
    if power == 4:
        basis[tail] = product * (
            6 * distance**2 + 4 * distance * (a + b) + a**2 + a * b + b**2
        )
    elif power == 3:
        basis[tail] = product * (3 * distance + a + b)
    elif power == 2:
        basis[tail] = product
    else:
        basis[tail] = 0.0
    return basis


def _basis(x, knots, norm, include_linear, kind, derivative):
    scale = {0: 1.0, 1: knots[-1] - knots[-2],
             2: (knots[-1] - knots[0])**(2 / 3)}[norm]
    degree = 4 if kind == "integral" else 3
    factor = prod(range(degree - derivative + 1, degree + 1)) / scale**derivative
    if kind == "integral":
        factor *= scale / 4
    nonlinear = factor * _nonlinear_basis(x, knots, scale, degree - derivative)
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)
    if kind == "integral":
        leading = (x, x**2 / 2) if derivative == 0 else ((ones, x) if derivative == 1 else (zeros, ones))
    else:
        leading = ((x,), (ones,), (zeros,))[derivative] if include_linear else ()
    return np.column_stack((*leading, nonlinear)) if leading else nonlinear


@dataclass(frozen=True)
class RestrictedCubicSpline:
    """A fitted basis transform that keeps training knots and rotation."""

    knots: tuple[float, ...]
    include_linear: bool
    norm: int
    kind: Literal["ordinary", "integral"]
    missing_value: float | None
    pc_center: tuple[float, ...] | None = None
    pc_scale: tuple[float, ...] | None = None
    pc_rotation: tuple[tuple[float, ...], ...] | None = None

    @property
    def nonlinear_columns(self) -> tuple[bool, ...]:
        """Mask of nonlinear columns, using the rms convention for PCA."""
        if self.pc_rotation is not None:
            return (True,) * len(self.pc_rotation[0])
        leading = (False, True) if self.kind == "integral" else ((False,) if self.include_linear else ())
        return leading + (True,) * (len(self.knots) - 2)

    def transform(self, x: ArrayLike, *, derivative: int = 0) -> NDArray[np.float64]:
        """Evaluate the saved basis or its first or second derivative."""
        if not isinstance(derivative, Integral) or derivative not in (0, 1, 2):
            raise ValueError("derivative must be 0, 1, or 2")
        values = _vector(x, "x")
        if self.missing_value is not None:
            values[np.isnan(values)] = self.missing_value
        design = _basis(values, np.asarray(self.knots), self.norm, self.include_linear, self.kind, derivative)
        if self.pc_rotation is not None:
            if derivative == 0:
                design -= np.asarray(self.pc_center)
            design = (design / np.asarray(self.pc_scale)) @ np.asarray(self.pc_rotation)
        design[np.isnan(values)] = np.nan
        return design


def fit_restricted_cubic_spline(
    x: ArrayLike,
    *,
    knots: ArrayLike | None = None,
    n_knots: int = 5,
    include_linear: bool = True,
    norm: Literal[0, 1, 2] = 2,
    kind: Literal["ordinary", "integral"] = "ordinary",
    missing_value: float | None = None,
    principal_components: bool = False,
    tie_fraction: float = 0.05,
) -> RestrictedCubicSpline:
    """Fit and save the spline design, or use explicitly supplied knots."""
    if not isinstance(norm, Integral) or norm not in (0, 1, 2):
        raise ValueError("norm must be 0, 1, or 2")
    if kind not in ("ordinary", "integral"):
        raise ValueError("kind must be 'ordinary' or 'integral'")
    if missing_value is not None and (not isinstance(missing_value, Real) or not np.isfinite(missing_value)):
        raise ValueError("missing_value must be None or a finite real number")
    if not isinstance(include_linear, (bool, np.bool_)) or not isinstance(principal_components, (bool, np.bool_)):
        raise ValueError("include_linear and principal_components must be boolean")
    values = _vector(x, "x")
    locations = (
        restricted_cubic_spline_knots(values, n_knots=n_knots, tie_fraction=tie_fraction)
        if knots is None
        else _check_knots(_vector(knots, "knots"))
    )
    configuration = (tuple(map(float, locations)), bool(include_linear), norm, kind, missing_value)
    fit = RestrictedCubicSpline(*configuration)
    if not principal_components:
        return fit
    design = fit.transform(values)
    if len(design) < 2 or not np.isfinite(design).all():
        raise ValueError("principal_components needs at least two finite training rows")
    center = design.mean(axis=0)
    scale = design.std(axis=0, ddof=1)
    if np.any(scale == 0):
        raise ValueError("principal_components requires nonconstant design columns")
    _, _, rotation_t = np.linalg.svd((design - center) / scale, full_matrices=False)
    rotation = rotation_t.T
    pivots = np.argmax(np.abs(rotation), axis=0)
    rotation *= np.sign(rotation[pivots, np.arange(rotation.shape[1])])
    return RestrictedCubicSpline(
        *configuration, tuple(center), tuple(scale), tuple(map(tuple, rotation)),
    )
