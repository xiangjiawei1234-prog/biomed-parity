"""Locally weighted regression on ordered observations."""

import numpy as np


def _anchor_positions(locations, delta):
    anchors = [0]
    last = len(locations) - 1
    while anchors[-1] < last:
        anchor = anchors[-1]
        within = np.searchsorted(locations, locations[anchor] + delta, side="right") - 1
        anchors.append(max(anchor + 1, within))
    return np.asarray(anchors, dtype=np.intp)


def _local_values(x, y, coordinates, first, neighbors, robust):
    left = 0
    scale = x[-1] - x[0]
    fitted = np.empty(len(coordinates))
    for i, center in enumerate(coordinates):
        while left + neighbors < len(x) and center - x[left] > x[left + neighbors] - center:
            left += 1
        radius = max(center - x[left], x[left + neighbors - 1] - center)
        if radius:
            end = np.searchsorted(x, np.nextafter(center + 0.999 * radius, np.inf), side="right")
            local = slice(left, end)
            relative = np.abs(x[local] - center) / radius
            weight = np.where(relative <= 0.001, 1.0, np.maximum(0, 1 - relative**3)**3)
            weight[relative > 0.999] = 0
        else:
            begin = np.searchsorted(x, center)
            end = np.searchsorted(x, center, side="right")
            local = slice(begin, end)
            weight = np.ones(end - begin)
        if robust is not None:
            weight *= robust[local]
        weight_sum = weight.sum()
        if weight_sum == 0:
            fitted[i] = y[first[i]]
            continue
        weight /= weight_sum
        offset = x[local] - center
        avg_offset = weight @ offset
        centered = offset - avg_offset
        variance = weight @ (centered * centered)
        if variance > (0.001 * scale)**2:
            weight *= 1 - avg_offset * centered / variance
        fitted[i] = weight @ y[local]
    return fitted


def _lowess(x, y, span, iterations, delta):
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    locations, first, inverse = np.unique(x, return_index=True, return_inverse=True)
    if len(x) < 2:
        return locations, y
    if delta is None:
        delta = 0.01 * (x[-1] - x[0])
    anchors = _anchor_positions(locations, delta)
    neighbors = min(len(x), max(2, int(span * len(x) + 1e-7)))
    robust = None
    for iteration in range(iterations + 1):
        values = _local_values(x, y, locations[anchors], first[anchors], neighbors, robust)
        fitted = np.interp(locations, locations[anchors], values)
        if iteration == iterations:
            break
        residual = np.abs(y - fitted[inverse])
        cutoff = 6 * np.median(residual)
        if cutoff < 1e-7 * residual.mean() or cutoff == 0:
            break
        relative = np.minimum(residual / cutoff, 1)
        robust = np.where(relative <= 0.001, 1.0, (1 - relative**2)**2)
        robust[relative > 0.999] = 0
    return locations, fitted
