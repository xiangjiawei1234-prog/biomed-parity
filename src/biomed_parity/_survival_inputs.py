"""Input conventions shared by right-censored survival methods."""

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _numeric_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if array.dtype.kind not in "biuf":
        if array.dtype.kind != "O" or not all(isinstance(x, Real) for x in array):
            raise ValueError(f"{name} must contain finite real numbers")
    result = array.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _event_vector(event: ArrayLike) -> NDArray[np.bool_]:
    event = _numeric_vector(event, "event")
    if not np.all((event == 0) | (event == 1)):
        raise ValueError("event must contain only 0 (censored) or 1 (event)")
    return event.astype(bool)


def _encode_labels(values, order, size, name):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or values.size != size:
        raise ValueError(f"{name} must be one-dimensional and match time")
    labels = values.tolist()
    if not (
        all(isinstance(label, str) for label in labels)
        or all(isinstance(label, Real) and np.isfinite(label) for label in labels)
    ):
        raise ValueError(f"{name} must contain strings or finite numeric labels of one kind")

    levels = tuple(sorted(set(labels))) if order is None else tuple(order)
    if len(levels) != len(set(labels)) or set(levels) != set(labels):
        raise ValueError(f"{name}_order must list each observed label exactly once")
    codes = {level: index for index, level in enumerate(levels)}
    return levels, np.fromiter((codes[label] for label in labels), dtype=np.intp, count=size)


def _fix_near_ties(time):
    times, inverse = np.unique(time, return_inverse=True)
    if times.size < 2:
        return time
    tolerance = np.sqrt(np.finfo(float).eps)
    magnitudes = np.abs(times)
    with np.errstate(over="ignore", divide="ignore"):
        gaps = np.diff(times)
        scale = magnitudes.mean()
        if not np.isfinite(scale):
            largest = magnitudes.max()
            scale = (magnitudes / largest).mean() * largest
        tied = (gaps <= tolerance) | (gaps / scale <= tolerance)
    if not tied.any():
        return time
    cut = np.r_[True, ~tied]
    return times[cut][np.cumsum(cut)[inverse] - 1]
