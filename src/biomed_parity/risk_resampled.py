"""Subject- and pair-averaged risk scores for shared validation resamples."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .risk_scores import _fit_censoring, _integrate_brier
from ._survival_inputs import _numeric_vector


@dataclass(frozen=True)
class ResampledRiskScores:
    """Leave-one-out Brier and leave-pair-out time-dependent AUC estimates."""

    horizons: NDArray[np.float64]
    auc: NDArray[np.float64]
    brier: NDArray[np.float64]
    integrated_brier: NDArray[np.float64]
    subject_validation_counts: NDArray[np.intp]
    unobserved_pairs: NDArray[np.intp]


def _pairwise_auc(risks, valid, cases, controls, weights):
    if not len(cases) or not len(controls):
        return np.full(risks.shape[1], np.nan), 0

    test_control = valid[:, controls].astype(np.int64)
    control_risks = risks[:, :, controls]
    control_weights = weights[controls]
    denominator = weights[cases].sum() * control_weights.sum()
    numerator = np.zeros(risks.shape[1])
    missing = 0
    # Process a block of case-control comparisons at a time.
    block_size = max(1, 1_000_000 // (len(valid) * len(controls)))
    for offset in range(0, len(cases), block_size):
        block = cases[offset:offset + block_size]
        count = valid[:, block].astype(np.int64).T @ test_control
        missing += int((count == 0).sum())
        pair_weights = np.divide(
            weights[block, None] * control_weights, count, where=count != 0,
            out=np.zeros(count.shape),
        )
        for model in range(risks.shape[1]):
            case = risks[:, model, block, None]
            control = control_risks[:, model, None, :]
            wins = np.count_nonzero(case > control, axis=0)
            ties = np.count_nonzero(case == control, axis=0)
            numerator[model] += np.sum((wins + ties / 2) * pair_weights)
    return np.clip(numerator / denominator, 0, 1), missing


def score_resampled_competing_risks(
    time: ArrayLike,
    status: ArrayLike,
    horizons: ArrayLike,
    risks: ArrayLike,
    *,
    cause: int = 1,
    auc_method: Literal["cause", "riskregression"] = "cause",
    censoring_covariates: ArrayLike | None = None,
    censoring_strata: ArrayLike | None = None,
) -> ResampledRiskScores:
    """Score shared resampled risks using the leave-one-/pair-out estimands.

    risks has shape resamples × models × subjects × horizons, or may omit
    the model axis for a single model. For each subject and resample give
    either complete probabilities or NaN everywhere. A subject's Brier
    residual is averaged over its validation draws. Case-control orderings
    are averaged over draws in which both subjects were held out together.
    """
    times = _numeric_vector(time, "time")
    events = _numeric_vector(status, "status")
    evaluation = _numeric_vector(horizons, "horizons")
    if times.shape != events.shape or np.any(times < 0):
        raise ValueError("time and status must be aligned with nonnegative follow-up")
    if np.any((events < 0) | (events != np.floor(events))):
        raise ValueError("status must contain nonnegative integer codes")
    if evaluation[0] < 0 or evaluation[-1] > times.max() or np.any(np.diff(evaluation) <= 0):
        raise ValueError("horizons must increase strictly within [0, maximum follow-up]")
    if not isinstance(cause, Integral) or isinstance(cause, bool) or cause < 1:
        raise ValueError("cause must be a positive integer")
    if auc_method not in ("cause", "riskregression"):
        raise ValueError("auc_method must be 'cause' or 'riskregression'")

    risks = np.asarray(risks)
    if risks.ndim == 3:
        risks = risks[:, None, :, :]
    if (
        risks.ndim != 4 or not all(risks.shape[:2]) or risks.shape[2:] != (len(times), len(evaluation))
        or risks.dtype.kind not in "buif" or np.any((risks < 0) | (risks > 1))
    ):
        raise ValueError("risks must contain probabilities or NaN by resample, model, subject and horizon")
    valid = np.isfinite(risks).all(axis=(1, 3))
    if not np.all(valid | np.isnan(risks).all(axis=(1, 3))) or not np.any(valid):
        raise ValueError("risks must use shared validation rows for all models and horizons")
    risks = risks.astype(float, copy=False)
    counts = valid.sum(axis=0)
    ever_valid = counts > 0
    censoring = _fit_censoring(times, events, censoring_covariates, censoring_strata)
    competing = (events > 0) & (events != cause)
    if auc_method == "riskregression":
        causes = np.unique(events[events > 0])
        if len(causes) > 1:
            competing = events == (causes[0] if cause == causes[1] else causes[1])
        # This reference keeps each subject's first prediction-horizon Wt
        # when reducing the response table to one row per subject.
        reference_weights, _ = censoring.weights_at(
            np.zeros(len(times), dtype=bool), times > evaluation[0], evaluation[0],
        )
    auc = np.empty((risks.shape[1], len(evaluation)))
    brier = np.empty_like(auc)
    unobserved = np.zeros(len(evaluation), dtype=np.intp)
    for index, horizon in enumerate(evaluation):
        after = times > horizon
        observed = ~after & (events != 0)
        cases = observed & (events == cause)
        controls = after | (observed & competing)
        weights, _ = censoring.weights_at(observed, after, horizon)
        errors = np.nansum((risks[..., index] - cases)**2, axis=0)
        errors = np.divide(errors, counts, where=ever_valid, out=np.zeros_like(errors))
        brier[:, index] = (errors @ weights) / len(times)
        auc_weights = np.where(after, reference_weights, weights) if auc_method == "riskregression" else weights
        auc[:, index], unobserved[index] = _pairwise_auc(
            risks[..., index], valid, np.flatnonzero(ever_valid & cases),
            np.flatnonzero(ever_valid & controls), auc_weights,
        )
    return ResampledRiskScores(
        evaluation, auc, brier, _integrate_brier(evaluation, brier), counts, unobserved,
    )
