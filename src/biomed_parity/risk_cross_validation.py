"""Out-of-fold risk scoring for competing-event prediction models."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _numeric_vector
from .risk_scores import CompetingRiskScores, score_competing_risks


RiskRefit = Callable[[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class CrossValidatedRiskScores:
    """One validation prediction per subject, followed by pooled score curves."""

    model_names: tuple[str, ...]
    folds: NDArray[np.intp]
    predictions: NDArray[np.float64]
    scores: CompetingRiskScores


def _fold_assignments(folds, n_folds, count, seed):
    if folds is None:
        n_folds = 5 if n_folds is None else n_folds
        if (not isinstance(n_folds, Integral) or isinstance(n_folds, (bool, np.bool_))
                or not 2 <= n_folds <= count):
            raise ValueError("n_folds must be an integer between 2 and the sample size")
        labels = np.arange(count, dtype=np.intp) % n_folds
        return np.random.default_rng(seed).permutation(labels)
    if n_folds is not None:
        raise ValueError("give explicit folds or n_folds")
    labels = np.asarray(folds)
    if (labels.shape != (count,) or labels.dtype.kind not in "iu"
            or len(np.unique(labels)) < 2 or np.any(labels < 0)
            or np.any(labels > np.iinfo(np.intp).max)):
        raise ValueError("folds must contain one nonnegative integer per subject and at least two distinct labels")
    return labels.astype(np.intp)


def _validation_inputs(time, status, horizons, refit_models):
    times = _numeric_vector(time, "time")
    events = _numeric_vector(status, "status")
    evaluation = _numeric_vector(horizons, "horizons")
    if times.shape != events.shape or np.any(times < 0):
        raise ValueError("time and status must be aligned with nonnegative follow-up")
    if np.any((events < 0) | (events != np.floor(events))):
        raise ValueError("status must contain nonnegative integer codes")
    if evaluation[0] < 0 or evaluation[-1] > times.max() or np.any(np.diff(evaluation) <= 0):
        raise ValueError("horizons must increase strictly within [0, maximum follow-up]")
    if not isinstance(refit_models, Mapping) or not refit_models or not all(
        isinstance(name, str) and name and callable(model) for name, model in refit_models.items()
    ):
        raise ValueError("refit_models must map nonempty model names to refitting functions")
    evaluation.flags.writeable = False
    return times, events, evaluation, tuple(refit_models.items())


def _validation_risks(model, train, held_out, evaluation, context):
    predicted = np.asarray(model(train, held_out, evaluation))
    if (
        predicted.shape != (len(held_out), len(evaluation)) or predicted.dtype.kind not in "buif"
        or not np.all((predicted >= 0) & (predicted <= 1))
    ):
        raise ValueError(f"{context}: expected one probability per held-out row and horizon")
    return predicted.astype(float, copy=False)


def cross_validate_competing_risks(
    time: ArrayLike,
    status: ArrayLike,
    horizons: ArrayLike,
    refit_models: Mapping[str, RiskRefit],
    *,
    folds: ArrayLike | None = None,
    n_folds: int | None = None,
    seed: int | None = None,
    cause: int = 1,
    confidence_level: float = 0.95,
    conservative: bool = False,
    censoring_covariates: ArrayLike | None = None,
    censoring_strata: ArrayLike | None = None,
    auc_inference: Literal["cause", "riskregression"] = "cause",
) -> CrossValidatedRiskScores:
    """Refit in each training fold and evaluate the pooled validation risks.

    A model callback receives training row indices, held-out indices, and
    evaluation horizons. It returns validation-subject × horizon cumulative
    incidences. All models share the same folds, which can be supplied as
    integer labels aligned with the original observations. Full-cohort IPCW
    weights are reused for the pooled out-of-fold predictions.
    """
    times, events, evaluation, model_items = _validation_inputs(time, status, horizons, refit_models)
    assignments = _fold_assignments(folds, n_folds, len(times), seed)
    predictions = np.empty((len(model_items), len(times), len(evaluation)))
    for fold in np.unique(assignments):
        train = np.flatnonzero(assignments != fold)
        held_out = np.flatnonzero(assignments == fold)
        train.flags.writeable = held_out.flags.writeable = False
        for index, (name, model) in enumerate(model_items):
            predictions[index, held_out] = _validation_risks(
                model, train, held_out, evaluation, f"model {name!r}, fold {fold}",
            )
    scores = score_competing_risks(
        times, events, evaluation, predictions, cause=cause, confidence_level=confidence_level,
        conservative=conservative, censoring_covariates=censoring_covariates,
        censoring_strata=censoring_strata, auc_inference=auc_inference,
    )
    return CrossValidatedRiskScores(tuple(name for name, _ in model_items), assignments, predictions, scores)
