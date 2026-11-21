import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.meta_influence import meta_influence
from biomed_parity.meta_multilevel import meta_multilevel
from biomed_parity.meta_multivariate import meta_multivariate


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "meta_multivariate"
CASES = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _inputs(case):
    levels = ("followup", "baseline", "discharge")
    study = np.repeat(np.arange(1, 21), 3)
    outcome = np.tile([1, 0, 2], 20)
    if case.get("incomplete"):
        keep = (3 * study + outcome + 1) % 11 != 0
        study = np.r_[study[keep], 3, 5]
        outcome = np.r_[outcome[keep], 1, 0]
    rows = np.arange(len(study))
    index = rows + 1
    moderator = (7 * index % 19) / 17
    variance = 0.052 + (11 * index % 13) / 300
    sine = np.sin(study * 1.24)
    cosine = np.cos(study * 0.92)
    random = np.column_stack((
        0.75 * sine + 0.37 * np.cos(study * 2.59),
        -0.51 * sine + 0.66 * cosine,
        0.45 * sine - 0.32 * cosine + 0.65 * np.sin(study * 2.17),
    ))
    if case["structure"] not in ("AR", "HAR"):
        random[:, 1] *= -1
    y = np.array([0.26, 0.55, -0.12])[outcome] + 0.36 * moderator
    y += random[rows, outcome] + 0.13 * np.cos(index * 2.8)
    V = 0.18 * np.outer(np.sqrt(variance), np.sqrt(variance)) * (study[:, None] == study)
    np.fill_diagonal(V, variance)
    X = np.column_stack((np.eye(3)[outcome], moderator))
    return y, V, study, np.asarray(levels)[outcome], levels, X


def _check(case, reference):
    y, V, study, endpoint, levels, X = _inputs(case)
    result = meta_multivariate(
        y, V, study, endpoint, outcome_levels=levels, structure=case["structure"],
        method=case["method"], test=case["test"], dfs=case.get("dfs", "residual"),
        moderators=X, intercept=False,
        tested_coefficients=[3], confidence_level=0.9,
    )
    assert result.outcome_levels == tuple(reference["levels"])
    assert result.structure == case["structure"]
    fit = result.fit
    for name, actual in {
        "random_covariance": result.random_covariance,
        "coefficients": fit.coefficients,
        "coefficient_covariance": fit.inference.covariance,
        "se": fit.inference.standard_errors,
        "p": fit.inference.p_values,
        "df": fit.inference.degrees_of_freedom,
        "confint": fit.inference.confidence_intervals,
        "likelihood": fit.log_likelihood,
        "omnibus": fit.omnibus.statistic,
        "omnibus_p": fit.omnibus.p_value,
        "qe": fit.heterogeneity.statistic,
    }.items():
        np.testing.assert_allclose(actual, np.asarray(reference[name], dtype=float),
                                   rtol=2e-5, atol=2e-7, err_msg=name)
    by_outcome = np.argmax(X[:, :3], axis=1)
    expected_marginal = V + result.random_covariance[by_outcome[:, None], by_outcome] * (
        study[:, None] == study
    )
    np.testing.assert_allclose(fit.marginal_covariance, expected_marginal, rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(result.random_correlation.diagonal(), 1, atol=2e-15)
    assert np.linalg.eigvalsh(result.random_covariance)[0] >= -1e-12


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_multivariate_meta_matches_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_multivariate_meta_matches_live_r():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    references = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, references[case["id"]])


def test_single_outcome_reduces_to_random_intercept():
    studies = np.repeat(np.arange(5), 2)
    y = np.array([-0.5, -0.4, 0.2, 0.25, 1.05, 1.2, 0.7, 0.75, -0.1, 0.08])
    V = np.linspace(0.1, 0.16, len(y))
    ref = meta_multilevel(y, V, random_intercepts={"studies": studies})
    result = meta_multivariate(y, V, studies, np.repeat("a", len(y)))
    np.testing.assert_allclose(result.random_covariance, [[ref.variance_components["studies"]]],
                               rtol=2e-7)
    np.testing.assert_allclose(result.fit.log_likelihood, ref.log_likelihood, atol=1e-12)
    with pytest.raises(ValueError, match="use reestimate=False"):
        meta_influence(result.fit, cluster=studies)
    diagnosis = meta_influence(result.fit, cluster=studies, reestimate=False)
    conditional_ref = meta_influence(ref, cluster=studies, reestimate=False)
    np.testing.assert_allclose(diagnosis.cooks_distance, conditional_ref.cooks_distance, rtol=2e-7)


def test_one_outcome_per_study_cannot_identify_unstructured_correlations():
    studies = np.arange(8)
    with pytest.raises(ValueError, match="not identifiable"):
        meta_multivariate(np.arange(8), np.ones(8), studies, np.tile(["a", "b"], 4))


def test_autoregression_requires_information_on_correlation_sign():
    outcomes = ["a", "c", "a", "c", "b", "b"]
    with pytest.raises(ValueError, match="observed odd lag"):
        meta_multivariate(
            np.arange(6), np.ones(6), [1, 1, 2, 2, 3, 4], outcomes,
            outcome_levels=["a", "b", "c"], structure="AR",
        )
