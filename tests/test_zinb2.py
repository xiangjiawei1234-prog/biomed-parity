import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.stats import nbinom

from biomed_parity.zero_inflated import fit_zinb2_random_intercept, zinb2_random_intercept_loglikelihood


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "zinb2"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _check(case, reference):
    data = np.asarray(case["observations"])
    n = len(data)
    design = {}
    if case["varied"]:
        for name, values in zip(("count", "inflation", "dispersion"), data[:, 2:5].T, strict=True):
            design[f"{name}_design"] = np.column_stack((np.ones(n), values))
        design["offset"] = data[:, 5]
    counts, groups = data[:, 0], data[:, 1]
    result = fit_zinb2_random_intercept(counts, groups, **design)
    assert result.converged, result.message
    for name in ("count", "inflation", "dispersion"):
        np.testing.assert_allclose(getattr(result, f"{name}_coefficients"), reference[name],
                                   atol=2e-6, rtol=3e-6)
    np.testing.assert_allclose(result.random_intercept_sd, reference["sd"], atol=2e-6, rtol=3e-6)
    np.testing.assert_allclose(result.log_likelihood, reference["log_likelihood"], atol=5e-8, rtol=0)
    parameters = np.array_split(case["trial"][:-1], 3)
    trial = zinb2_random_intercept_loglikelihood(
        counts, groups, *parameters, np.exp(case["trial"][-1]), **design,
    )
    np.testing.assert_allclose(trial, reference["trial_log_likelihood"], atol=5e-8, rtol=0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_zinb2_glmmtmb_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_zinb2_live_glmmtmb():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    reference = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, reference[case["id"]])


def test_zinb2_likelihood_reduces_to_independent_mixture_at_zero_random_variance():
    counts = np.array([0, 1, 0, 12, 6, 2, 0])
    means = np.array([2, 4, 0.7, 6, 4, 2, 1.0])
    group = [0, 0, 1, 2, 1, 2, 2]
    shape = np.array([0.55, 0.9, 0.5, 2.2, 3.0, 4, 2])
    eta = np.log(shape)[:, None]
    zero_chance = 0.24
    predicted = (1 - zero_chance) * nbinom.pmf(counts, shape, shape / (shape + means))
    predicted += zero_chance * (counts == 0)
    actual = zinb2_random_intercept_loglikelihood(
        counts, group, [0], [np.log(zero_chance / (1 - zero_chance))], [1], 0,
        offset=np.log(means), dispersion_design=eta,
    )
    np.testing.assert_allclose(actual, np.log(predicted).sum(), atol=2e-14, rtol=0)
