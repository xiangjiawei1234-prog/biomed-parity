import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.interval_regression import fit_interval_regression


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "interval_regression"
CASES = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _inputs(case):
    i = np.arange(1, 41)
    x = ((i * 7) % 13 - 6) / 4
    z = (i * 5 % 11 > 5).astype(float)
    lp = -0.36 * x + (0.22 * z if case.get("two_covariates") else 0)
    latent = np.exp(1.1 + lp + 0.48 * (np.sin(i * 1.7) + np.cos(i * 0.93)))
    left = np.floor(latent * 2) / 2
    right = left + 0.5
    right[left >= 4.5] = np.inf
    left[left >= 4.5] = 4.5
    left[right <= 1.5] = 0
    exact = (i % 7 == 0) & (left > 0) & (left < 4.5)
    left[exact] = right[exact] = latent[exact]
    weights = 0.5 + (i * 11 % 9) / 6 if case.get("weighted") else None
    design = np.column_stack((x, z)) if case.get("two_covariates") else x
    return left, right, design, weights


def _check(case, expected):
    left, right, design, weights = _inputs(case)
    fit = fit_interval_regression(
        left, right, design, model=case["model"],
        bounds=case.get("bounds", (False, True)), weights=weights,
    )
    intervals = [[np.inf if value is None else value for value in row]
                 for row in expected["intervals"]]
    np.testing.assert_allclose(fit.baseline.intervals, intervals, rtol=0, atol=2e-10)
    np.testing.assert_allclose(fit.coefficients, expected["coefficients"], rtol=0, atol=3e-5)
    np.testing.assert_allclose(fit.log_likelihood, expected["log_likelihood"], rtol=0, atol=2e-8)
    query = [0.3, 1] if case.get("two_covariates") else [0.3]
    distributions = (fit.baseline, fit.distribution(query))
    survival = np.column_stack([np.r_[1, 1 - np.cumsum(dist.mass)] for dist in distributions])
    np.testing.assert_allclose(survival, expected["survival"], rtol=0, atol=8e-6)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_interval_regression_matches_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_interval_regression_matches_live_r():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    outputs = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, outputs[case["id"]])


def test_zero_weights_and_translation_of_covariates():
    case = CASES[1]
    left, right, design, _ = _inputs(case)
    reference = fit_interval_regression(left, right, design, model="po")
    recentered = fit_interval_regression(
        np.r_[left, 0.1], np.r_[right, 0.1], np.r_[design + 4, 6],
        weights=np.r_[np.ones(len(design)), 0], model="po",
    )
    np.testing.assert_array_equal(recentered.baseline.intervals, reference.baseline.intervals)
    np.testing.assert_allclose(recentered.coefficients, reference.coefficients, rtol=1e-7)
    np.testing.assert_allclose(recentered.distribution([4]).mass, reference.baseline.mass, atol=3e-8)
    with pytest.raises(ValueError, match="full-rank design"):
        fit_interval_regression(left, right, np.ones_like(design))
