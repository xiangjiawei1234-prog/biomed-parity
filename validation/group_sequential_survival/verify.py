"""Compare native trial projections and designs with R gsDesign."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.group_sequential_survival import (
    SurvivalTrial,
    expected_survival_events,
    fixed_survival_design,
    group_sequential_survival_design,
    survival_event_time,
)

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    trial = SurvivalTrial(**case.get("trial", {}))
    options = case.get("options", {})
    sequential = case["kind"] == "sequential"

    def compare(field, actual, expected):
        if not sequential:
            atol, rtol = 1e-8, 1e-10
        elif field in {"analysis_times", "minimum_followup", "accrual_durations"}:
            atol, rtol = 3e-5, 0
        else:
            atol, rtol = 7e-7, 5e-7
        np.testing.assert_allclose(
            np.ravel(actual), np.ravel(expected), atol=atol, rtol=rtol,
            err_msg=f"{case['id']} {field}",
        )

    def compare_counts(actual, expected):
        for field, value in expected.items():
            compare(field, getattr(actual, field), value)

    if case["kind"] == "calendar":
        counts = [expected_survival_events(trial, time) for time in case["times"]]
        for actual, expected in zip(counts, reference["counts"], strict=True):
            compare_counts(actual, expected)
        target = counts[-1].total_events * case["event_fraction"]
        actual_time = survival_event_time(trial, target, max_time=case["times"][-1])
        compare("event_time", actual_time, reference["event_time"])
        return

    if sequential:
        design = group_sequential_survival_design(case["fractions"], trial=trial, **options)
        compare("fixed_events", design.fixed_design.counts.total_events, reference["fixed_events"])
        compare("event target recovery", design.counts.total_events, design.event_targets)
        compare("analysis_times", design.analysis_times, reference["analysis_times"])
        for field in ("event_targets", "upper_boundaries", "lower_boundaries", "alternative_effect"):
            if reference[field] is None:
                assert np.isneginf(getattr(design, field)).all()
            else:
                compare(field, getattr(design, field), reference[field])
    else:
        design = fixed_survival_design(trial, **options)
        compare("analysis_time", design.analysis_time, reference["analysis_time"])
    compare_counts(design.counts, reference["counts"])
    for field in ("minimum_followup", "power"):
        compare(field, getattr(design, field), reference[field])
    for field in ("accrual_durations", "accrual_rates"):
        compare(field, getattr(design.trial, field), reference[field])


def verify(rscript="Rscript"):
    proc = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    reference = {case["id"]: case for case in report["cases"]}
    assert set(reference) == {case["id"] for case in CASES}
    for case in CASES:
        assert_case(case, reference[case["id"]])
    return {"method": "gsDesign::nSurv/gsSurv/eEvents", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))
