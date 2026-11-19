from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.competing_calibration import calibrate_competing_risks, jackknife_cumulative_incidence


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "competing_calibration"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _check_case(case, expected):
    args = {key: value for key, value in case.items() if key != "id"}
    if args.pop("pseudo_only", False):
        actual = {"pseudo_values": jackknife_cumulative_incidence(**args)}
    else:
        actual = asdict(calibrate_competing_risks(**args))
    assert actual.keys() == expected.keys()
    for field, value in actual.items():
        if value is None:
            assert expected[field] is None
        else:
            reference = np.asarray(expected[field], dtype=float)
            np.testing.assert_allclose(value, reference, rtol=2e-12, atol=8e-13, err_msg=field)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_r_calibration(case):
    _check_case(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    output = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    expected = json.loads(output.stdout)["results"]
    for case in CASES:
        _check_case(case, expected[case["id"]])


def test_aggregated_jackknife_matches_separate_leave_one_out_cohorts():
    rng = np.random.default_rng(93181)
    for _ in range(20):
        time = rng.integers(0, 10, 20)
        status = rng.integers(0, 4, len(time))
        h = 5
        cause = 3

        def incidence(t, d):
            survival = 1.0
            cumulative = 0.0
            for knot in np.unique(t[t <= h]):
                risk = np.count_nonzero(t >= knot)
                deaths = d[t == knot]
                cumulative += survival * np.count_nonzero(deaths == cause) / risk
                survival *= 1 - np.count_nonzero(deaths > 0) / risk
            return cumulative

        full = incidence(time, status)
        separate = [len(time) * full - (len(time) - 1) * incidence(
            np.delete(time, j), np.delete(status, j),
        ) for j in range(len(time))]
        np.testing.assert_allclose(jackknife_cumulative_incidence(time, status, h, cause=cause),
                                   separate, rtol=1e-11, atol=2e-13)


def test_tied_quantile_groups_keep_outcomes_aligned_to_predictions():
    args = [[1, 2, 3, 4, 5, 6], [1, 1, 2, 2, 1, 2], [.8, .2, .8, .2, .8, .2], 4]
    first = calibrate_competing_risks(*args, groups=5)
    other = calibrate_competing_risks(*[arr[::-1] for arr in args[:3]], 4, groups=5)
    np.testing.assert_allclose(first.predicted, [.2, .8])
    np.testing.assert_allclose(first.observed, [1 / 3, 1 / 3])
    np.testing.assert_array_equal(first.observations, [3, 3])
    np.testing.assert_allclose(first.observed, other.observed)


@pytest.mark.parametrize("settings,message", [
    ({"horizon": 4}, "horizon"),
    ({"risks": [1, 1.1, 1]}, "probability"),
    ({"method": "nne"}, "bandwidth"),
    ({"method": "invalid"}, "method must"),
    ({"groups": [.4, .5]}, "span"),
])
def test_inputs(settings, message):
    args = dict(time=[1, 2, 3], status=[1, 0, 2], risks=[.1, .4, .8], horizon=2)
    with pytest.raises(ValueError, match=message):
        calibrate_competing_risks(**(args | settings))
