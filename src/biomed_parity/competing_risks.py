"""Array adapters for selected cmprsk outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cmprsk_timepoints(
    curve_times: ArrayLike,
    estimates: ArrayLike,
    variances: ArrayLike,
    times: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a cumulative-incidence step curve with R cmprsk conventions.

    Return sorted unique query times, incidence estimates, and variances.
    Repeated knots store values immediately before and after a jump. At a
    jump, return the post-jump value. Queries outside the curve return NaN.
    NaN query times are omitted and infinite query times raise ValueError.
    """
    knots = np.asarray(curve_times, dtype=float)
    est = np.asarray(estimates, dtype=float)
    var = np.asarray(variances, dtype=float)
    query = np.atleast_1d(np.asarray(times, dtype=float))
    if knots.ndim != 1 or knots.size == 0 or est.shape != knots.shape or var.shape != knots.shape:
        raise ValueError("curve times, estimates, and variances must be nonempty aligned vectors")
    if not np.isfinite(knots).all() or np.any(knots[1:] < knots[:-1]):
        raise ValueError("curve times must be finite and sorted in nondecreasing order")
    if query.ndim != 1 or np.isinf(query).any():
        raise ValueError("query times must be a vector without infinities")

    query = np.unique(query[~np.isnan(query)])
    indices = np.searchsorted(knots, query, side="right") - 1
    within = (indices >= 0) & (query <= knots[-1])
    output_est = np.full(query.shape, np.nan)
    output_var = np.full(query.shape, np.nan)
    output_est[within] = est[indices[within]]
    output_var[within] = var[indices[within]]
    return query, output_est, output_var


def cmprsk_residuals_by_time(
    event_times: ArrayLike,
    event_residuals: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sum Fine–Gray event residuals at each unique time of a target event.

    Pass one time per residual row from the complete cases used by the fit,
    with both inputs in corresponding order. Only target-event rows belong
    in the inputs. The returned times are ascending.
    """
    times = np.asarray(event_times, dtype=float)
    residuals = np.asarray(event_residuals, dtype=float)
    if times.ndim != 1 or residuals.ndim != 2 or residuals.shape[0] != times.size:
        raise ValueError("event times must have one entry per residual row")
    if not np.isfinite(times).all():
        raise ValueError("event times must be finite")
    if times.size == 0:
        return times.copy(), residuals.copy()

    order = np.argsort(times, kind="stable")
    unique_times, first_indices = np.unique(times[order], return_index=True)
    return unique_times, np.add.reduceat(residuals[order], first_indices, axis=0)
