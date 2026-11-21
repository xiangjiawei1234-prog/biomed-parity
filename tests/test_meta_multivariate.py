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
    levels = (0, 7.25, 2.5) if case["structure"] == "CAR" else ("followup", "baseline", "discharge")
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
    if "extra" in case:
        y += 0.65 * np.cos((3 * study % 7) * 2.31)
    V = 0.18 * np.outer(np.sqrt(variance), np.sqrt(variance)) * (study[:, None] == study)
    np.fill_diagonal(V, variance)
    X = np.column_stack((np.eye(3)[outcome], moderator))
    return y, V, study, np.asarray(levels)[outcome], levels, X


def _check(case, reference):
    y, V, study, endpoint, levels, X = _inputs(case)
    site = 3 * study % 7
    groups = {"site": site} if "extra" in case else None
    fixed = {"site": 0.2} if case.get("extra") == "fixed" else None
    result = meta_multivariate(
        y, V, study, endpoint, outcome_levels=levels, structure=case["structure"],
        method=case["method"], test=case["test"], dfs=case.get("dfs", "residual"),
        moderators=X, intercept=False,
        random_intercepts=groups, fixed_variances=fixed,
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
    if groups:
        np.testing.assert_allclose(fit.variance_components["site"], reference["site_variance"],
                                   rtol=2e-5, atol=2e-7)
        expected_marginal += fit.variance_components["site"] * (site[:, None] == site)
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


def test_continuous_time_rejects_non_numeric_outcomes_and_allows_independence():
    studies = np.repeat(np.arange(10), 2)
    outcomes = [0, 1.5] * 10
    first = np.linspace(-0.8, 0.8, 10)
    effects = np.column_stack((first, -first)).ravel()
    variance = np.full(len(effects), 0.05)
    result = meta_multivariate(effects, variance, studies, outcomes, structure="CAR")
    ref = meta_multivariate(effects, variance, studies, outcomes, structure="ID")
    np.testing.assert_allclose(result.random_covariance, ref.random_covariance, rtol=2e-7)
    assert result.random_correlation[0, 1] == 0
    with pytest.raises(ValueError, match="CAR outcome times"):
        meta_multivariate(effects, variance, studies, ["a", "b"] * 10, structure="CAR")


def test_additional_known_kernel_and_inestimable_study_intercept():
    case = next(case for case in CASES if case["id"] == "unstructured-and-random-intercept")
    y, V, studies, outcomes, levels, X = _inputs(case)
    site = 3 * studies % 7
    kwargs = dict(outcome_levels=levels, moderators=X, intercept=False)
    with pytest.raises(ValueError, match="not identifiable"):
        meta_multivariate(y, V, studies, outcomes, random_intercepts={"study": studies}, **kwargs)
    result = meta_multivariate(
        y, V, studies, outcomes, covariance_components={"site": site[:, None] == site}, **kwargs,
    )
    ref = meta_multivariate(y, V, studies, outcomes, random_intercepts={"site": site}, **kwargs)
    np.testing.assert_array_equal(result.random_covariance, ref.random_covariance)
    assert result.fit.variance_components == ref.fit.variance_components
