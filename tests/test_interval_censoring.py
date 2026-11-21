import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.interval_censoring import turnbull


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "interval_censoring"
CASES = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _fit(case):
    right = [np.inf if value is None else value for value in case["right"]]
    return turnbull(case["left"], right, bounds=case.get("bounds", (False, True)),
                    weights=case.get("weights"))


def _check(case, reference):
    fit = _fit(case)
    intervals = [[np.inf if value is None else value for value in row]
                 for row in reference["intervals"]]
    # icenReg shifts open endpoints inward by 1e-10 before identifying the support.
    np.testing.assert_allclose(fit.intervals, intervals, rtol=0, atol=2e-10)
    np.testing.assert_allclose(fit.mass, reference["mass"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(fit.log_likelihood, reference["log_likelihood"], rtol=0, atol=5e-12)
    assert fit.optimality_gap <= 1e-7


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_turnbull_matches_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_turnbull_matches_live_r():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    outputs = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, outputs[case["id"]])


def test_survival_bounds_respect_exact_and_open_lower_endpoints():
    fit = turnbull([1, 1, 3], [1, 2, np.inf])
    np.testing.assert_array_equal(fit.intervals, [[1, 1], [1, 2], [3, np.inf]])
    np.testing.assert_array_equal(fit.closed, [[True, True], [False, True], [False, False]])
    np.testing.assert_allclose(
        fit.survival_bounds([0, 1, 1.5, 2, 3, 4, np.inf]),
        [[1, 1], [2/3, 2/3], [1/3, 2/3], [1/3, 1/3], [1/3, 1/3], [0, 1/3], [0, 0]],
    )
    np.testing.assert_allclose(fit.survival_bounds(1.5), [1/3, 2/3])


def test_open_intervals_use_literal_endpoint_order_without_time_shift():
    adjacent = 2.0 + 1e-12
    fit = turnbull([1, 2], [2, adjacent], bounds=(False, False))
    np.testing.assert_array_equal(fit.intervals, [[1, 2], [2, adjacent]])
    np.testing.assert_array_equal(fit.mass, [0.5, 0.5])
    np.testing.assert_array_equal(fit.survival_bounds(2), [0.5, 0.5])


def test_zero_weights_can_neither_create_nor_split_support():
    fit = turnbull([0, 1, 2], [3, 1, 2], weights=[1, 0, 0])
    np.testing.assert_array_equal(fit.intervals, [[0, 3]])
    np.testing.assert_array_equal(fit.mass, [1])
    assert fit.log_likelihood == 0


@pytest.mark.parametrize("kwargs, message", [
    ({"bounds": (0, np.nan)}, "bounds"),
    ({"weights": [0, 0]}, "weights"),
    ({"weights": [-1, 1]}, "weights"),
    ({"max_iterations": 0}, "max_iterations"),
    ({"tolerance": 0.0}, "tolerance"),
])
def test_turnbull_rejects_malformed_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        turnbull([0, 2], [3, np.inf], **kwargs)
