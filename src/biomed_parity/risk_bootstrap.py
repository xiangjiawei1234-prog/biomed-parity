"""Out-of-bag bootstrap validation metrics for competing-risk predictions."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .risk_cross_validation import RiskRefit, _validation_inputs, _validation_risks
from .risk_scores import _fit_censoring, _integrate_brier, _weighted_auc


@dataclass(frozen=True)
class BootstrapRiskSummary:
    """Mean, replicate SD and percentile interval across out-of-bag scores."""

    estimate: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    replicates: NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapRiskContrast:
    """Paired score differences between two models in each resample."""

    model: int
    reference: int
    auc: BootstrapRiskSummary
    brier: BootstrapRiskSummary


@dataclass(frozen=True)
class BootstrapRiskValidation:
    """Out-of-bag results from a common series of refitted training samples."""

    model_names: tuple[str, ...]
    horizons: NDArray[np.float64]
    training_indices: NDArray[np.intp]
    out_of_bag: NDArray[np.bool_]
    auc: BootstrapRiskSummary
    brier: BootstrapRiskSummary
    integrated_brier: BootstrapRiskSummary
    contrasts: tuple[BootstrapRiskContrast, ...]
    predictions: NDArray[np.float64] | None


def _resamples(train_indices, sample_size, training_size, n_resamples, replace, seed):
    if train_indices is not None:
        indices = np.asarray(train_indices)
        if (
            indices.ndim != 2 or not all(indices.shape) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= sample_size)
        ):
            raise ValueError("train_indices must be a nonempty matrix of in-range row indices")
        if training_size is not None:
            raise ValueError("give train_indices or training_size")
        return indices.astype(np.intp)
    count = sample_size if training_size is None else training_size
    if (not isinstance(count, Integral) or isinstance(count, bool) or count <= 0
            or (not replace and count >= sample_size)):
        raise ValueError("training_size must be positive and leave validation rows without replacement")
    if not isinstance(n_resamples, Integral) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(sample_size, size=count, replace=replace) for _ in range(n_resamples)])


def _bootstrap_summary(replicates, level):
    alpha = (1 - level) / 2
    outputs = np.full((4, *replicates.shape[1:]), np.nan)
    for index in np.ndindex(replicates.shape[1:]):
        values = replicates[(slice(None), *index)]
        values = values[np.isfinite(values)]
        if len(values):
            outputs[(0, *index)] = values.mean()
            outputs[(2, *index)], outputs[(3, *index)] = np.quantile(values, [alpha, 1 - alpha])
            if len(values) > 1:
                outputs[(1, *index)] = values.std(ddof=1)
    return BootstrapRiskSummary(*outputs, replicates)


def bootstrap_validate_competing_risks(
    time: ArrayLike,
    status: ArrayLike,
    horizons: ArrayLike,
    refit_models: Mapping[str, RiskRefit],
    *,
    train_indices: ArrayLike | None = None,
    n_resamples: int = 200,
    training_size: int | None = None,
    replace: bool = True,
    seed: int | np.random.Generator | None = None,
    cause: int = 1,
    confidence_level: float = 0.95,
    censoring_covariates: ArrayLike | None = None,
    censoring_strata: ArrayLike | None = None,
    keep_predictions: bool = False,
) -> BootstrapRiskValidation:
    """Fit each model in each training sample and score its out-of-bag predictions.

    Arrays of scores are indexed by resample, model and horizon. Missing
    scores are omitted from the bootstrap summaries. Use train_indices for
    fixed index rows, preserving multiplicity. Censoring weights use the
    full original cohort and are restricted to each validation sample.
    """
    times, events, evaluation, models = _validation_inputs(time, status, horizons, refit_models)
    if not isinstance(cause, Integral) or isinstance(cause, bool) or cause < 1:
        raise ValueError("cause must be a positive integer")
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if not all(isinstance(flag, (bool, np.bool_)) for flag in (replace, keep_predictions)):
        raise ValueError("replace and keep_predictions must be boolean")
    if len(times) < 2:
        raise ValueError("bootstrap validation requires at least two subjects")
    training = _resamples(train_indices, len(times), training_size, n_resamples, replace, seed)
    out_of_bag = np.ones((len(training), len(times)), dtype=bool)
    np.put_along_axis(out_of_bag, training, False, axis=1)
    if not out_of_bag.any():
        raise ValueError("no resample leaves any out-of-bag subjects")
    censoring = _fit_censoring(times, events, censoring_covariates, censoring_strata)
    case = (times[:, None] <= evaluation) & (events[:, None] == cause)
    weights = np.column_stack([
        censoring.weights_at((times <= h) & (events != 0), times > h, h)[0] for h in evaluation
    ])

    auc = np.full((len(training), len(models), len(evaluation)), np.nan)
    brier = np.full_like(auc, np.nan)
    integrated_brier = np.full_like(auc, np.nan)
    predictions = (
        np.full((len(training), len(models), len(times), len(evaluation)), np.nan)
        if keep_predictions else None
    )
    training.flags.writeable = False
    for replicate, train in enumerate(training):
        held_out = np.flatnonzero(out_of_bag[replicate])
        if not len(held_out):
            continue
        held_out.flags.writeable = False
        cases = case[held_out]
        weight = weights[held_out]
        for model_index, (name, refit) in enumerate(models):
            predicted = _validation_risks(
                refit, train, held_out, evaluation, f"model {name!r}, resample {replicate}",
            )
            if predictions is not None:
                predictions[replicate, model_index, held_out] = predicted
            brier[replicate, model_index] = np.mean(weight * (predicted - cases)**2, axis=0)
            for j, (score, case_group, weight_at_horizon) in enumerate(zip(
                predicted.T, cases.T, weight.T, strict=True,
            )):
                auc[replicate, model_index, j] = _weighted_auc(
                    score, weight_at_horizon * case_group, weight_at_horizon * ~case_group,
                )[0]
        integrated_brier[replicate] = _integrate_brier(evaluation, brier[replicate])
    contrasts = tuple(
        BootstrapRiskContrast(
            model, reference,
            _bootstrap_summary(auc[:, model] - auc[:, reference], confidence_level),
            _bootstrap_summary(brier[:, model] - brier[:, reference], confidence_level),
        )
        for reference, model in combinations(range(len(models)), 2)
    )
    return BootstrapRiskValidation(
        tuple(name for name, _ in models), evaluation, training, out_of_bag,
        _bootstrap_summary(auc, confidence_level), _bootstrap_summary(brier, confidence_level),
        _bootstrap_summary(integrated_brier, confidence_level), contrasts, predictions,
    )
