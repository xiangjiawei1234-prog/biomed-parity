import json
import os

import numpy as np
import pytest
from scipy.stats import norm

from biomed_parity.group_sequential import crossing_probabilities
from biomed_parity.group_sequential_survival import (
    SurvivalTrial,
    expected_survival_events,
    fixed_survival_design,
    group_sequential_survival_design,
    survival_event_time,
)
from validation.group_sequential_survival.verify import CASES, HERE, assert_case, verify

REFERENCE = {
    case["id"]: case
    for case in json.loads((HERE / "expected.json").read_text(encoding="utf-8"))["cases"]
}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_r_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_r_reference():
    assert verify()["cases_passed"] == len(CASES)


def test_constant_rate_projection_and_inversion():
    trial = SurvivalTrial(
        control_hazards=0.2, control_dropout=0.05, hazard_ratio=0.5,
        experimental_dropout=0.1, accrual_rates=20, accrual_durations=4,
    )
    time = 8
    counts = expected_survival_events(trial, time)
    for event_hazard, dropout_hazard, events in (
        (0.2, 0.05, counts.control_events), (0.1, 0.1, counts.experimental_events),
    ):
        hazard = event_hazard + dropout_hazard
        survival_area = (np.exp(-4 * hazard) - np.exp(-8 * hazard)) / hazard
        assert events[0] == pytest.approx(10 * event_hazard / hazard * (4 - survival_area))
    assert counts.total_enrollment == pytest.approx(80)
    assert survival_event_time(trial, counts.total_events, max_time=10) == pytest.approx(time)
    assert survival_event_time(trial, 0, max_time=10) == 0
    assert expected_survival_events(trial, 0).total_events == 0
    with pytest.raises(ValueError, match="exceeds the projected events"):
        survival_event_time(trial, counts.total_events + 1, max_time=time)


def test_zero_hazard_period_and_tiny_event_rate():
    trial = SurvivalTrial(control_hazards=[0, 1e-12], hazard_durations=[1], accrual_durations=10)
    assert expected_survival_events(trial, 1).total_events == 0
    events = expected_survival_events(trial, 2).total_events
    assert events == pytest.approx(0.4e-12, rel=1e-10, abs=0)


def test_high_experimental_dropout_applies_at_each_look():
    assumptions = SurvivalTrial(control_dropout=0.01, experimental_dropout=0.2, allocation_ratio=2)
    design = group_sequential_survival_design(trial=assumptions, test_type=1)
    assert design.counts.control_events.shape == (3, 1)
    assert design.counts.experimental_events[0, 0] < design.counts.control_events[0, 0]
    np.testing.assert_allclose(design.counts.total_events, design.event_targets, atol=1e-10, rtol=0)
    np.testing.assert_allclose(design.counts.experimental_enrollment, 2 * design.counts.control_enrollment)
    crossing = crossing_probabilities(
        design.information_fractions, design.lower_boundaries, design.upper_boundaries,
        effect=design.alternative_effect,
    )
    assert crossing.upper_total == pytest.approx(0.9, abs=1e-12)


def test_single_fixed_analysis_with_null_effect_matches_alpha():
    trial = SurvivalTrial(hazard_ratio=1)
    design = group_sequential_survival_design([1], trial=trial, beta=None)
    assert design.power == pytest.approx(0.025)
    assert design.upper_boundaries[0] == pytest.approx(norm.isf(0.025))
    assert design.alternative_effect == pytest.approx(0)


@pytest.mark.parametrize(
    ("trial", "options", "message"),
    [
        (SurvivalTrial(hazard_ratio=1), {}, "must differ"),
        (SurvivalTrial(null_hazard_ratio=1.2), {"method": "Schoenfeld"}, "requires null_hazard_ratio"),
        (SurvivalTrial(control_hazards=[[0.08, 0.1]]), {"method": "Freedman"}, "single stratum"),
        (SurvivalTrial(accrual_rates=0), {}, "positive event information"),
        (SurvivalTrial(), {"solve": "followup_duration"}, "target cannot be reached"),
        (SurvivalTrial(), {"final_time": 5, "minimum_followup": 6}, "must exceed"),
    ],
)
def test_invalid_fixed_design(trial, options, message):
    with pytest.raises(ValueError, match=message):
        fixed_survival_design(trial, **options)


def test_invalid_rate_shapes_and_durations():
    with pytest.raises(ValueError, match="match"):
        SurvivalTrial(control_hazards=[0.1, 0.3])
    with pytest.raises(ValueError, match="positive durations"):
        SurvivalTrial(accrual_durations=[4, 0])
