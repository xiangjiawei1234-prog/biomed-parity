"""Event counts and calendar design for proportional-hazards survival trials."""

from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from .group_sequential import (
    _information_vector,
    crossing_probabilities,
    efficacy_boundaries,
    efficacy_futility_boundaries,
    symmetric_boundaries,
)


def _nonnegative_array(values, name, dimensions):
    array = np.asarray(values)
    if (
        array.ndim not in dimensions or not array.size or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all() or np.any(array < 0)
    ):
        raise ValueError(f"{name} must contain finite nonnegative numbers with the required shape")
    return array.astype(float)


def _durations(values, name):
    array = _nonnegative_array(values, name, (0, 1)).reshape(-1)
    if np.any(array == 0) or not np.isfinite(array.sum()):
        raise ValueError(f"{name} must contain positive durations with a finite sum")
    return array


def _rate_matrix(values, name, rows, strata):
    array = _nonnegative_array(values, name, (0, 1, 2))
    if array.ndim < 2:
        array = array.reshape(-1, 1)
    try:
        return np.broadcast_to(array, (rows, strata))
    except ValueError:
        raise ValueError(f"{name} must match the number of periods and strata") from None


def _scalar(value, name, *, positive):
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        or not np.isfinite(value) or value < 0 or (positive and value == 0)
    ):
        limit = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {limit} number")
    return float(value)


@dataclass(frozen=True)
class SurvivalTrial:
    """Two-arm trial rates in a common unit of time.

    Rows of rate matrices are periods and columns are strata. A vector
    specifies successive periods in one stratum or a schedule shared by
    all strata. A one-row matrix specifies rates by stratum. The last
    event and dropout period extends indefinitely after enrollment.
    """

    control_hazards: ArrayLike = np.log(2) / 6
    hazard_ratio: float = 0.6
    null_hazard_ratio: float = 1.0
    control_dropout: ArrayLike = 0.0
    experimental_dropout: ArrayLike | None = None
    accrual_rates: ArrayLike = 1.0
    accrual_durations: ArrayLike = 12.0
    hazard_durations: ArrayLike = ()
    allocation_ratio: float = 1.0

    def __post_init__(self):
        accrual = _durations(self.accrual_durations, "accrual_durations")
        hazard = np.asarray(self.hazard_durations)
        if hazard.shape != (0,):
            hazard = _durations(hazard, "hazard_durations")
        control = _nonnegative_array(self.control_hazards, "control_hazards", (0, 1, 2))
        strata = control.shape[1] if control.ndim == 2 else 1
        hazard_periods = len(hazard) + 1
        control = _rate_matrix(control, "control_hazards", hazard_periods, strata)
        dropout = _rate_matrix(self.control_dropout, "control_dropout", hazard_periods, strata)
        experimental_dropout = dropout if self.experimental_dropout is None else _rate_matrix(
            self.experimental_dropout, "experimental_dropout", hazard_periods, strata
        )
        rates = _rate_matrix(self.accrual_rates, "accrual_rates", len(accrual), strata)
        for name in ("hazard_ratio", "null_hazard_ratio", "allocation_ratio"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, positive=True))
        for name, array in (
            ("control_hazards", control), ("control_dropout", dropout),
            ("experimental_dropout", experimental_dropout), ("accrual_rates", rates),
            ("accrual_durations", accrual), ("hazard_durations", hazard),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def accrual_duration(self) -> float:
        return float(self.accrual_durations.sum())


@dataclass(frozen=True)
class SurvivalTrialCounts:
    """Continuous expected counts by stratum, optionally by analysis and stratum."""

    control_events: NDArray[np.float64]
    experimental_events: NDArray[np.float64]
    control_enrollment: NDArray[np.float64]
    experimental_enrollment: NDArray[np.float64]

    @property
    def total_events(self) -> float | NDArray[np.float64]:
        return (self.control_events + self.experimental_events).sum(axis=-1)

    @property
    def total_enrollment(self) -> float | NDArray[np.float64]:
        return (self.control_enrollment + self.experimental_enrollment).sum(axis=-1)


@dataclass(frozen=True)
class FixedSurvivalDesign:
    """Fixed analysis design with its fitted accrual schedule."""

    trial: SurvivalTrial
    counts: SurvivalTrialCounts
    analysis_time: float
    minimum_followup: float
    power: float
    method: str
    solve: str


@dataclass(frozen=True)
class GroupSequentialSurvivalDesign:
    """Calendar forecasts and efficacy/futility boundaries at every analysis."""

    trial: SurvivalTrial
    fixed_design: FixedSurvivalDesign
    counts: SurvivalTrialCounts
    information_fractions: NDArray[np.float64]
    analysis_times: NDArray[np.float64]
    event_targets: NDArray[np.float64]
    upper_boundaries: NDArray[np.float64]
    lower_boundaries: NDArray[np.float64]
    alternative_effect: float
    minimum_followup: float
    power: float
    solve: str


def _expected_counts(trial, time, control_hazards, experimental_hazards):
    # With analysis at time t, enrollment at calendar time u has age t - u.
    # In age order, cut the integral at every hazard and reversed accrual break.
    accrual_ends = np.cumsum(trial.accrual_durations)
    hazard_ends = np.cumsum(trial.hazard_durations)
    ages = np.unique(np.r_[0, time, hazard_ends[hazard_ends < time], time - accrual_ends[accrual_ends < time]])
    widths = np.diff(ages)[:, None]
    midpoints = ages[:-1] + widths[:, 0] / 2
    hazard_periods = np.searchsorted(hazard_ends, midpoints)
    accrual_periods = np.searchsorted(accrual_ends, time - midpoints)
    accrual_rates = np.vstack((trial.accrual_rates, np.zeros(trial.accrual_rates.shape[1])))
    accrual = accrual_rates[accrual_periods]
    allocation = trial.allocation_ratio
    control_fraction = 1 / (1 + allocation)
    experimental_fraction = allocation / (1 + allocation)

    def events(hazards, dropout):
        failure = hazards[hazard_periods]
        total_hazard = failure + dropout[hazard_periods]
        rate_share = np.divide(failure, total_hazard, out=np.zeros_like(failure), where=total_hazard > 0)
        cumulative_hazard = total_hazard * widths
        alive = np.exp(-np.vstack((np.zeros(failure.shape[1]), np.cumsum(cumulative_hazard, axis=0)[:-1])))
        risk = -np.expm1(-cumulative_hazard)
        incidence = alive * rate_share * risk
        prior_incidence = np.vstack((np.zeros(failure.shape[1]), np.cumsum(incidence, axis=0)[:-1]))
        # Average incidence within each period. The expansion avoids cancellation
        # when the combined failure and dropout hazard times the span is small.
        small = cumulative_hazard < 1e-4
        x = cumulative_hazard[small]
        incidence_fraction = np.ones_like(risk)
        incidence_fraction[~small] -= risk[~small] / cumulative_hazard[~small]
        incidence_fraction[small] = x * (1 / 2 - x * (1 / 6 - x * (1 / 24 - x / 120)))
        average_incidence = prior_incidence + alive * rate_share * incidence_fraction
        return (accrual * widths * average_incidence).sum(axis=0)

    if time == 0:
        zero = np.zeros(accrual_rates.shape[1])
        return SurvivalTrialCounts(zero.copy(), zero.copy(), zero.copy(), zero)
    enrolled = (accrual * widths).sum(axis=0)
    return SurvivalTrialCounts(
        events(control_hazards, trial.control_dropout) * control_fraction,
        events(experimental_hazards, trial.experimental_dropout) * experimental_fraction,
        enrolled * control_fraction, enrolled * experimental_fraction,
    )


def expected_survival_events(trial: SurvivalTrial, analysis_time: float) -> SurvivalTrialCounts:
    """Project enrollment and events at a calendar time under the alternative."""
    time = _scalar(analysis_time, "analysis_time", positive=False)
    return _expected_counts(trial, time, trial.control_hazards, trial.control_hazards * trial.hazard_ratio)


def survival_event_time(trial: SurvivalTrial, target_events: float, *, max_time: float) -> float:
    """Find the time at which the trial is expected to have the target events."""
    target = _scalar(target_events, "target_events", positive=False)
    end = _scalar(max_time, "max_time", positive=True)
    if target == 0:
        return 0.0
    if expected_survival_events(trial, end).total_events < target:
        raise ValueError("target_events exceeds the projected events at max_time")
    return brentq(lambda time: expected_survival_events(trial, time).total_events - target, 0, end)


_METHODS = ("LachinFoulkes", "Schoenfeld", "Freedman", "BernsteinLagakos")
_SOLVES = ("accrual_rate", "accrual_duration", "followup_duration")


def _design_options(trial, minimum_followup, final_time, alpha, beta, sided, method, solve):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if solve not in _SOLVES:
        raise ValueError(f"solve must be one of {_SOLVES}")
    if beta is not None and trial.hazard_ratio == trial.null_hazard_ratio:
        raise ValueError("hazard_ratio and null_hazard_ratio must differ for sizing")
    if method in ("Schoenfeld", "Freedman") and trial.null_hazard_ratio != 1:
        raise ValueError(f"{method} requires null_hazard_ratio = 1")
    if method == "Freedman" and trial.control_hazards.shape[1] > 1:
        raise ValueError("Freedman requires a single stratum")
    if not isinstance(sided, Integral) or isinstance(sided, bool) or sided not in (1, 2):
        raise ValueError("sided must be 1 or 2")
    alpha = _scalar(alpha, "alpha", positive=True) / sided
    if alpha >= 0.5:
        raise ValueError("alpha per side must be less than 0.5")
    if beta is not None:
        beta = _scalar(beta, "beta", positive=True)
        if beta >= 0.5:
            raise ValueError("beta must be less than 0.5")
    elif solve != "accrual_rate":
        raise ValueError("beta is required when solving for a duration")
    if final_time is not None and solve != "accrual_rate":
        raise ValueError("final_time is determined when solving for a duration")
    if solve == "followup_duration":
        if minimum_followup is not None:
            raise ValueError("minimum_followup is determined when solving for followup_duration")
        return trial, None, None, alpha, beta
    if final_time is None:
        followup = 6.0 if minimum_followup is None else _scalar(minimum_followup, "minimum_followup", positive=False)
        return trial, followup, trial.accrual_duration + followup, alpha, beta
    final = _scalar(final_time, "final_time", positive=True)
    followup = max(0, final - trial.accrual_duration) if minimum_followup is None else _scalar(
        minimum_followup, "minimum_followup", positive=False
    )
    accrual = final - followup
    if accrual <= 0:
        raise ValueError("final_time must exceed minimum_followup")
    if (
        minimum_followup is not None and len(trial.accrual_durations) > 1
        and trial.accrual_duration - accrual > np.sqrt(np.finfo(float).eps) * max(1, accrual)
    ):
        raise ValueError("accrual_durations exceeds final_time - minimum_followup")
    return _with_accrual_duration(trial, accrual), followup, final, alpha, beta


def _with_accrual_duration(trial, duration):
    ends = np.r_[np.cumsum(trial.accrual_durations[:-1]), np.inf]
    lengths = np.diff(np.r_[0, ends[ends < duration], duration])
    return replace(trial, accrual_durations=lengths, accrual_rates=trial.accrual_rates[:len(lengths)])


def _with_accrual_scale(trial, scale):
    return replace(trial, accrual_rates=trial.accrual_rates * scale)


def _event_information(counts, *, pool_strata):
    control, experimental = counts.control_events, counts.experimental_events
    if pool_strata:
        control, experimental = control.sum(), experimental.sum()
    total = control + experimental
    # A stratum with no events in either arm contributes no information.
    return float(np.sum(control * np.divide(experimental, total, out=np.zeros_like(total), where=total > 0)))


def _signal(trial, counts, time, method, *, pool_strata):
    hr, hr0, allocation = trial.hazard_ratio, trial.null_hazard_ratio, trial.allocation_ratio
    delta = abs(np.log(hr) - np.log(hr0))
    if method == "Schoenfeld":
        info = counts.total_events * (1 / (1 + allocation)) * (allocation / (1 + allocation))
        return delta, info, info
    if method == "Freedman":
        delta = abs(hr - 1) / (hr + 1 / allocation)
        info = counts.total_events / allocation
        return delta, info, info
    null_control = trial.control_hazards
    if method == "LachinFoulkes":
        null_control = null_control * ((1 + hr * allocation) / (1 + hr0 * allocation))
    null_counts = _expected_counts(trial, time, null_control, null_control * hr0)
    return delta, _event_information(null_counts, pool_strata=pool_strata), _event_information(
        counts, pool_strata=pool_strata
    )


def _effect_at_schedule(trial, time, method, z_alpha, *, pool_strata=False):
    counts = expected_survival_events(trial, time)
    delta, null_info, alternative_info = _signal(trial, counts, time, method, pool_strata=pool_strata)
    if min(null_info, alternative_info) <= 0:
        return counts, -np.inf, np.inf
    signal = delta * np.sqrt(alternative_info)
    threshold = z_alpha * np.sqrt(alternative_info / null_info)
    return counts, signal, threshold


def _duration_root(objective, name):
    # Start near zero to keep the enrollment schedule strictly positive.
    lower, upper = 1e-8, 10000
    if objective(lower) > 0:
        raise ValueError(f"target already exceeded at the minimum {name}")
    if objective(upper) < 0:
        raise ValueError(f"target cannot be reached within 10000 time units of {name}")
    return brentq(objective, lower, upper, xtol=1e-10)


def _fixed_design(trial, followup, time, alpha, beta, method, solve):
    z_alpha = -ndtri(alpha)
    if solve == "accrual_rate":
        counts, signal, threshold = _effect_at_schedule(trial, time, method, z_alpha)
        if beta is None:
            return FixedSurvivalDesign(trial, counts, time, followup, float(ndtr(signal - threshold)), method, "power")
        if signal <= 0:
            raise ValueError("sizing requires positive event information under both hypotheses")
        scale = ((threshold - ndtri(beta)) / signal) ** 2
        trial = _with_accrual_scale(trial, scale)
    else:
        z_beta = -ndtri(beta)

        def objective(candidate, candidate_time):
            _, signal, threshold = _effect_at_schedule(
                candidate, candidate_time, method, z_alpha, pool_strata=method == "LachinFoulkes"
            )
            return signal - threshold - z_beta

        if solve == "accrual_duration":
            duration = _duration_root(
                lambda duration: objective(_with_accrual_duration(trial, duration), duration + followup), "accrual"
            )
            trial = _with_accrual_duration(trial, duration)
        else:
            followup = _duration_root(
                lambda duration: objective(trial, trial.accrual_duration + duration), "follow-up"
            )
        time = trial.accrual_duration + followup
    counts = expected_survival_events(trial, time)
    return FixedSurvivalDesign(trial, counts, time, followup, 1 - beta, method, solve)


def fixed_survival_design(
    trial: SurvivalTrial | None = None,
    *,
    minimum_followup: float | None = None,
    final_time: float | None = None,
    alpha: float = 0.025,
    beta: float | None = 0.1,
    sided: int = 1,
    method: str = "LachinFoulkes",
    solve: str = "accrual_rate",
) -> FixedSurvivalDesign:
    """Size a fixed-analysis trial or evaluate achieved power with beta=None.

    The default scales accrual rates proportionately. Set solve to
    'accrual_duration' or 'followup_duration' to preserve accrual rates
    while solving for one duration. When both time arguments are omitted,
    the minimum follow-up is six time units unless it is being solved.
    """
    trial = SurvivalTrial() if trial is None else trial
    trial, followup, time, alpha, beta = _design_options(
        trial, minimum_followup, final_time, alpha, beta, sided, method, solve
    )
    return _fixed_design(trial, followup, time, alpha, beta, method, solve)


def _sequential_boundaries(fractions, alpha, beta, test_type, spending, parameter, futility_spending, futility_parameter):
    if test_type == 1:
        design = efficacy_boundaries(fractions, alpha=alpha, spending=spending, parameter=parameter)
        upper = design.z_boundaries
        lower = np.full(len(fractions), -np.inf)
    elif test_type == 2:
        design = symmetric_boundaries(fractions, alpha=alpha, spending=spending, parameter=parameter)
        upper, lower = design.upper_boundaries, design.lower_boundaries
    else:
        design = efficacy_futility_boundaries(
            fractions, alpha=alpha, beta=beta, spending=spending, parameter=parameter,
            futility_spending=futility_spending, futility_parameter=futility_parameter, binding=test_type == 3,
        )
        return design.lower_boundaries, design.upper_boundaries, design.alternative_effect

    def objective(effect):
        upper_power = crossing_probabilities(fractions, lower, upper, effect=effect).upper_total
        return upper_power - (1 - beta)

    effect = brentq(objective, 0, 2 * (-ndtri(alpha) - ndtri(beta)) + 12)
    return lower, upper, effect


def group_sequential_survival_design(
    information_fractions: ArrayLike = (1 / 3, 2 / 3, 1.0),
    *,
    trial: SurvivalTrial | None = None,
    minimum_followup: float | None = None,
    final_time: float | None = None,
    alpha: float = 0.025,
    beta: float | None = 0.1,
    sided: int = 1,
    method: str = "LachinFoulkes",
    solve: str = "accrual_rate",
    test_type: int = 4,
    spending: str = "hsd",
    parameter: float | None = None,
    futility_spending: str = "hsd",
    futility_parameter: float | None = None,
) -> GroupSequentialSurvivalDesign:
    """Size an event-driven trial and project enrollment, events and analysis times.

    Fractions are cumulative proportions of the planned final event count.
    The design types are one-sided efficacy (1), symmetric efficacy (2),
    binding futility (3) and nonbinding futility (4). The power target is
    crossing the upper boundary under the design alternative.
    """
    trial = SurvivalTrial() if trial is None else trial
    fractions = _information_vector(information_fractions)
    if not isinstance(test_type, Integral) or isinstance(test_type, bool) or test_type not in (1, 2, 3, 4):
        raise ValueError("test_type must be an integer from 1 to 4")
    if len(fractions) > 1 and beta is None:
        raise ValueError("beta is required for multiple analyses")
    trial, followup, time, alpha, beta = _design_options(
        trial, minimum_followup, final_time, alpha, beta, sided, method, solve
    )
    if len(fractions) == 1:
        fixed = _fixed_design(trial, followup, time, alpha, beta, method, solve)
        trial, followup, time = fixed.trial, fixed.minimum_followup, fixed.analysis_time
        upper = efficacy_boundaries(fractions, alpha=alpha, spending=spending, parameter=parameter).z_boundaries
        lower = np.array([-np.inf])
        if beta is None:
            _, signal, threshold = _effect_at_schedule(trial, time, method, upper[0])
            effect = upper[0] + signal - threshold
        else:
            effect = upper[0] - ndtri(beta)
        targets = np.array([fixed.counts.total_events])
    else:
        lower, upper, effect = _sequential_boundaries(
            fractions, alpha, beta, test_type, spending, parameter, futility_spending, futility_parameter
        )
        z_alpha = -ndtri(alpha)
        inflation = (effect / (z_alpha - ndtri(beta))) ** 2
        if solve == "followup_duration":
            # The reference fixed design is sized at the sequential final time.
            def objective(duration):
                _, signal, threshold = _effect_at_schedule(trial, trial.accrual_duration + duration, method, z_alpha)
                return signal / np.sqrt(inflation) - threshold + ndtri(beta)

            followup = _duration_root(objective, "follow-up")
            time = trial.accrual_duration + followup
            fixed = _fixed_design(trial, followup, time, alpha, beta, method, "accrual_rate")
        else:
            fixed = _fixed_design(trial, followup, time, alpha, beta, method, solve)
        targets = fractions * (fixed.counts.total_events * inflation)
        if solve == "accrual_rate":
            trial = _with_accrual_scale(fixed.trial, inflation)
        elif solve == "accrual_duration":
            def objective(duration):
                candidate = _with_accrual_duration(trial, duration)
                return expected_survival_events(candidate, duration + followup).total_events - targets[-1]

            duration = _duration_root(objective, "accrual")
            trial = _with_accrual_duration(trial, duration)
            time = trial.accrual_duration + followup

    times = np.r_[[survival_event_time(trial, target, max_time=time) for target in targets[:-1]], time]
    rows = [expected_survival_events(trial, look_time) for look_time in times]
    counts = SurvivalTrialCounts(*(
        np.stack([getattr(row, field) for row in rows]) for field in SurvivalTrialCounts.__dataclass_fields__
    ))
    return GroupSequentialSurvivalDesign(
        trial, fixed, counts, fractions, times, targets, upper, lower, float(effect), followup,
        fixed.power, fixed.solve if len(fractions) == 1 else solve,
    )
