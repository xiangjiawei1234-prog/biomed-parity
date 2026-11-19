"""IPCW scores for fixed-horizon competing-risk predictions."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ._censoring import _CensoringCox
from ._survival_inputs import _encode_labels, _numeric_vector


@dataclass(frozen=True)
class ScoreSummary:
    estimate: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    influence: NDArray[np.float64]


@dataclass(frozen=True)
class ScoreContrast:
    model: int
    reference: int
    auc: ScoreSummary
    brier: ScoreSummary
    auc_pvalues: NDArray[np.float64]
    brier_pvalues: NDArray[np.float64]


@dataclass(frozen=True)
class CompetingRiskScores:
    horizons: NDArray[np.float64]
    auc: ScoreSummary
    brier: ScoreSummary
    contrasts: tuple[ScoreContrast, ...]
    integrated_brier: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class _CensoringKM:
    knots: np.ndarray
    indices: np.ndarray
    at_risk: np.ndarray
    failure: np.ndarray
    censored: np.ndarray
    survival: np.ndarray

    @classmethod
    def fit(cls, time, censored):
        knots, indices, counts = np.unique(time, return_inverse=True, return_counts=True)
        at_risk = len(time) + counts - counts.cumsum()
        failures = np.bincount(indices, weights=censored, minlength=len(knots))
        censor_risk = at_risk - counts + failures
        rate = np.divide(failures, censor_risk, where=censor_risk > 0,
                         out=np.zeros_like(failures))
        survival = np.r_[1.0, np.cumprod(1 - rate)]
        return cls(knots, indices, at_risk, failures, censored, survival)

    def sampling_influence(self, sensitivity, included):
        """Propagate derivative contributions of the fitted inverse weights."""
        n_knots = len(self.knots)
        last_knot = np.bincount(included + 1, weights=sensitivity, minlength=n_knots + 1)[1:]
        tail = np.cumsum(last_knot[::-1])[::-1]
        jump = tail / self.at_risk
        drift = np.cumsum(jump * self.failure / self.at_risk)
        return (self.censored * jump[self.indices]) - drift[self.indices]

    def weights_at(self, observed_event, after_horizon, horizon):
        known = observed_event | after_horizon
        as_of = np.searchsorted(self.knots, horizon, side="right") - 1
        included = np.where(observed_event, self.indices - 1, np.where(after_horizon, as_of, -1))
        weight = np.divide(
            known, self.survival[included + 1], where=known, out=np.zeros(len(known)),
        )
        return weight, included

    def hazard_influence_at(self, horizon):
        """Influence of the marginal censoring cumulative hazard at one horizon."""
        at_horizon = np.searchsorted(self.knots, horizon, side="right") - 1
        if at_horizon < 0:
            return np.zeros(len(self.indices))
        prefix = np.cumsum(self.failure / self.at_risk**2)
        event = self.censored * (self.indices <= at_horizon) / self.at_risk[self.indices]
        drift = prefix[np.minimum(self.indices, at_horizon)]
        return len(self.indices) * (event - drift)


def _fit_censoring(times, events, covariates, strata):
    if covariates is None and strata is None:
        return _CensoringKM.fit(times, events == 0)

    size = len(times)
    if covariates is None:
        design = np.empty((size, 0))
    else:
        design = np.asarray(covariates)
        if design.ndim == 1:
            design = design[:, None]
        if (
            design.ndim != 2 or len(design) != size or not design.shape[1]
            or design.dtype.kind not in "buif" or not np.isfinite(design).all()
        ):
            raise ValueError("censoring_covariates must contain finite sample-aligned columns")
        design = design.astype(float)
    codes = np.zeros(size, dtype=int) if strata is None else _encode_labels(
        strata, None, size, "censoring_strata",
    )[1]
    return _CensoringCox.fit(times, events == 0, design, codes)


def _integrate_brier(horizons, scores):
    interval_widths = np.diff(np.r_[0.0, horizons])
    left_scores = np.column_stack((np.zeros(len(scores)), scores[:, :-1]))
    areas = np.cumsum(left_scores * interval_widths, axis=1)
    return np.divide(areas, horizons, out=np.zeros_like(areas), where=horizons > 0)


def _weighted_auc(prediction, case_weight, control_weight, *, reported_auc=None):
    cases = case_weight.sum()
    controls = control_weight.sum()
    if cases == 0 or controls == 0:
        return np.nan, np.full(len(prediction), np.nan)

    _, order = np.unique(prediction, return_inverse=True)
    case_mass = np.bincount(order, weights=case_weight)
    control_mass = np.bincount(order, weights=control_weight)
    control_lower = np.cumsum(control_mass) - control_mass / 2
    case_higher = np.cumsum(case_mass[::-1])[::-1] - case_mass / 2
    case_numerator = case_weight * control_lower[order]
    control_numerator = control_weight * case_higher[order]
    pair_total = cases * controls
    auc = float(np.clip(case_numerator.sum() / pair_total, 0, 1))
    relative_cases = case_weight / cases
    relative_controls = control_weight / controls
    reference = auc if reported_auc is None else reported_auc
    sensitivity = (
        (case_numerator + control_numerator) / pair_total
        - reference * (relative_cases + relative_controls)
    )
    return auc, sensitivity


def _riskregression_auc_influence(
    prediction, weight, observed_event, after_horizon, event_order, reported_auc,
    censoring, included, horizon, last_horizon, conservative,
):
    # The legacy reference estimates the point AUC for the requested cause.
    # Its influence routine uses the first cause as cases and the second as
    # observed competing controls, even when the point estimate used another cause.
    first = observed_event & (event_order == 1)
    control = after_horizon | (observed_event & (event_order == 2))
    first_auc, sensitivity = _weighted_auc(
        prediction, weight * first, weight * control, reported_auc=reported_auc,
    )
    influence = len(weight) * sensitivity
    if not conservative and horizon < last_horizon:
        influence += censoring.sampling_influence(influence, included)
        if isinstance(censoring, _CensoringKM):
            # The KM path initializes both numerator tails with the reported AUC.
            influence += 2 * (reported_auc - first_auc) * censoring.hazard_influence_at(horizon)
    return influence


def _summary(estimate, influence, critical, *, clipped):
    sample_size = influence.shape[-1]
    centered = influence - influence.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.linalg.norm(centered, axis=-1) / np.sqrt(sample_size * (sample_size - 1))
    lower, upper = estimate - critical * se, estimate + critical * se
    if clipped:
        lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
    return ScoreSummary(estimate, se, lower, upper, influence)


def _pvalues(summary):
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = summary.estimate / summary.standard_error
    return 2 * norm.sf(np.abs(statistic))


def score_competing_risks(
    time: ArrayLike,
    status: ArrayLike,
    horizons: ArrayLike,
    risks: ArrayLike,
    *,
    cause: int = 1,
    confidence_level: float = 0.95,
    conservative: bool = False,
    censoring_covariates: ArrayLike | None = None,
    censoring_strata: ArrayLike | None = None,
    auc_inference: Literal["cause", "riskregression"] = "cause",
) -> CompetingRiskScores:
    """Score out-of-sample cumulative-incidence probabilities by horizon.

    Status 0 denotes right censoring. Cases experience the target event by
    the horizon. Controls include observed competing events and subjects
    still event free at the horizon. Each risk matrix is subjects × horizons.
    Pass models × subjects × horizons for paired model contrasts.

    Optional censoring covariates or strata select a reverse Cox model with
    Efron ties. Otherwise use marginal reverse Kaplan–Meier. Integrated Brier
    scores use the riskRegression left-step convention starting at zero.
    Select `auc_inference="riskregression"` to reproduce the historical
    influence calculation from riskRegression 2026.03.11.
    """
    times = _numeric_vector(time, "time")
    events = _numeric_vector(status, "status")
    evaluation = _numeric_vector(horizons, "horizons")
    if times.shape != events.shape or np.any(times < 0):
        raise ValueError("time and status must be aligned with nonnegative follow-up")
    if np.any((events < 0) | (events != np.floor(events))):
        raise ValueError("status must contain nonnegative integer codes")
    if not isinstance(cause, (int, np.integer)) or isinstance(cause, bool) or cause < 1:
        raise ValueError("cause must be a positive integer")
    if np.any(evaluation < 0) or np.any(evaluation > times.max()) or np.any(np.diff(evaluation) <= 0):
        raise ValueError("horizons must increase strictly within [0, maximum follow-up]")
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if not isinstance(conservative, (bool, np.bool_)):
        raise ValueError("conservative must be a boolean")
    if auc_inference not in ("cause", "riskregression"):
        raise ValueError("auc_inference must be 'cause' or 'riskregression'")

    predictions = np.asarray(risks)
    if predictions.ndim == 2:
        predictions = predictions[None, :, :]
    if (
        predictions.ndim != 3 or not len(predictions)
        or predictions.shape[1:] != (len(times), len(evaluation))
        or predictions.dtype.kind not in "buif"
        or not np.all((predictions >= 0) & (predictions <= 1))
    ):
        raise ValueError("risks must contain one probability per model, subject and horizon")

    censoring = _fit_censoring(times, events, censoring_covariates, censoring_strata)
    causes = np.unique(events[events > 0])
    event_order = np.where(events > 0, np.searchsorted(causes, events) + 1, 0)
    last_horizon = times.max()
    n_models, n_subjects, n_horizons = predictions.shape
    auc = np.empty((n_models, n_horizons))
    brier = np.empty_like(auc)
    auc_influence = np.empty((n_models, n_horizons, n_subjects))
    brier_influence = np.empty_like(auc_influence)
    for horizon_index, horizon in enumerate(evaluation):
        case = (times <= horizon) & (events == cause)
        observed_event = (times <= horizon) & (events != 0)
        after_horizon = times > horizon
        weight, included = censoring.weights_at(observed_event, after_horizon, horizon)
        for model, prediction in enumerate(predictions[:, :, horizon_index]):
            brier_rows = weight * (prediction - case) ** 2
            brier[model, horizon_index] = brier_rows.mean()
            error_if = brier_rows - brier[model, horizon_index]
            area, sensitivity = _weighted_auc(prediction, weight * case, weight * ~case)
            auc[model, horizon_index] = area
            area_if = n_subjects * sensitivity
            if not conservative:
                error_if += censoring.sampling_influence(brier_rows, included)
            if auc_inference == "riskregression":
                area_if = _riskregression_auc_influence(
                    prediction, weight, observed_event, after_horizon, event_order, area,
                    censoring, included, horizon, last_horizon, conservative,
                )
            elif not conservative:
                area_if += censoring.sampling_influence(area_if, included)
            auc_influence[model, horizon_index] = area_if
            brier_influence[model, horizon_index] = error_if

    critical = norm.isf((1 - confidence_level) / 2)
    contrasts = []
    for reference, model in combinations(range(n_models), 2):
        delta_auc = _summary(
            auc[model] - auc[reference], auc_influence[model] - auc_influence[reference],
            critical, clipped=False,
        )
        delta_brier = _summary(
            brier[model] - brier[reference], brier_influence[model] - brier_influence[reference],
            critical, clipped=False,
        )
        contrasts.append(ScoreContrast(
            model, reference, delta_auc, delta_brier, _pvalues(delta_auc), _pvalues(delta_brier),
        ))
    return CompetingRiskScores(
        evaluation,
        _summary(auc, auc_influence, critical, clipped=True),
        _summary(brier, brier_influence, critical, clipped=True),
        tuple(contrasts),
        _integrate_brier(evaluation, brier),
    )
