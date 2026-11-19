"""Validate multilevel covariance estimation and robust inference against R."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.meta_analysis import meta_reml
from biomed_parity.meta_multilevel import meta_cluster_robust, meta_multilevel


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "meta_multilevel"
SPEC = json.loads((FIXTURES / "cases.json").read_text())
REFERENCE = json.loads((FIXTURES / "expected.json").read_text())


def _inputs(case):
    data = SPEC["datasets"][case["dataset"]]
    v = np.asarray(data["variances"])
    codes = np.asarray(data["sampling_clusters"])
    V = data["sampling_correlation"] * np.sqrt(v[:, None] * v) * (codes[:, None] == codes)
    np.fill_diagonal(V, v)
    terms = dict(case.get("groups", data.get("groups", {})))
    known_covariances = {}
    for name, correlation in case.get("known_correlations", {}).items():
        _, codes = np.unique(terms.pop(name), return_inverse=True)
        known_covariances[name] = np.asarray(correlation)[np.ix_(codes, codes)]

    options = {name: case[name] for name in (
        "method", "test", "dfs", "confidence_level", "intercept", "fixed_variances",
        "profile", "tested_coefficients",
    ) if name in case}
    return data, V, dict(
        moderators=data["moderators"], random_intercepts=terms,
        covariance_components=known_covariances, **options,
    )


def _compare(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key, value in expected.items():
            _compare(actual[key], value, f"{label}.{key}")
    else:
        expected = np.asarray(expected, dtype=float)
        if label.endswith(".degrees_of_freedom") and expected.shape == (1,):
            expected = np.broadcast_to(expected, np.shape(actual))
        # With crossed random effects the reference's numerical optimizer
        # stops slightly before the analytic likelihood score reaches zero.
        rtol = 1.5e-5 if label.startswith("crossed-reml.") else 3e-6
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float), expected, rtol=rtol, atol=4e-8, err_msg=label,
        )


def _check_fixture(case, reference):
    data, V, options = _inputs(case)
    fit = meta_multilevel(data["effects"], V, **options)
    actual = asdict(fit)
    actual.update(actual.pop("inference"))
    exclude = {"method", "confidence_level", "_design", "_tested"}
    if not case.get("profile"):
        exclude.add("variance_intervals")
    actual = {name: value for name, value in actual.items() if name not in exclude}
    if not reference.get("variance_components"):
        reference = {**reference, "variance_components": {}}
    if "robust" in reference:
        actual["robust"] = {}
        for correction, expected in reference["robust"].items():
            robust = meta_cluster_robust(fit, data["robust_clusters"], correction=correction)
            assert robust.cluster_count == len(set(data["robust_clusters"]))
            actual["robust"][correction] = dict(asdict(robust.inference), omnibus=asdict(robust.omnibus))
    _compare(actual, reference, case["id"])


@pytest.mark.parametrize("case", SPEC["cases"], ids=lambda case: case["id"])
def test_exact_metafor_reference(case):
    _check_fixture(case, REFERENCE["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1")
def test_live_metafor_reference():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=45,
    )
    live = json.loads(process.stdout)
    for case in SPEC["cases"]:
        _check_fixture(case, live["results"][case["id"]])


def test_independent_random_intercepts_reduce_to_simple_reml():
    y = [.2, .75, .47, -.08, .31]
    variances = [.03, .06, .025, .05, .07]
    independent = meta_reml(y, variances, test="knha")
    multilevel = meta_multilevel(
        y, variances, random_intercepts={"effect": list(range(len(y)))}, test="knha", profile=True,
    )
    np.testing.assert_allclose(multilevel.variance_components["effect"], independent.tau2, rtol=2e-6)
    np.testing.assert_allclose(multilevel.coefficients[0], independent.mean_effect, rtol=2e-7)
    np.testing.assert_allclose(
        multilevel.variance_intervals["effect"], independent.tau2_confidence_interval, atol=1e-7,
    )
    np.testing.assert_allclose(
        multilevel.inference.confidence_intervals[0], independent.confidence_interval, atol=2e-8,
    )


def test_cr2_and_fit_are_invariant_to_observation_order():
    case = SPEC["cases"][0]
    data, V, original_options = _inputs(case)
    options = {**original_options, "profile": False}
    y = np.asarray(data["effects"])
    groups = np.asarray(data["robust_clusters"])
    permutation = np.random.default_rng(122).permutation(len(y))
    analytical = meta_multilevel(y, V, **options)
    reorder = {
        **options, "moderators": np.asarray(options["moderators"])[permutation],
        "random_intercepts": {
            name: np.asarray(labels)[permutation] for name, labels in options["random_intercepts"].items()
        },
    }
    permuted = meta_multilevel(y[permutation], V[np.ix_(permutation, permutation)], **reorder)
    one = meta_cluster_robust(analytical, groups)
    two = meta_cluster_robust(permuted, groups[permutation])
    np.testing.assert_allclose(permuted.coefficients, analytical.coefficients, rtol=1e-8)
    np.testing.assert_allclose(two.inference.covariance, one.inference.covariance, rtol=6e-8)
    np.testing.assert_allclose(two.omnibus.p_value, one.omnibus.p_value, rtol=2e-7)


def test_singular_sampling_covariance_can_be_regularized_by_model():
    y = [.14,.38,.08,.27,.47]
    fit = meta_multilevel(y, [0.0] * len(y), random_intercepts={"effect": list(range(len(y)))})
    assert fit.variance_components["effect"] == pytest.approx(np.var(y, ddof=1), rel=2e-7)
    assert np.isnan(fit.heterogeneity.statistic)
    assert np.all(np.linalg.eigvalsh(fit.marginal_covariance) > 0)


def test_balanced_cr2_matches_cluster_mean_t_test():
    effect = [.07,.13,.15,.25,.25,.35,.33,.47]
    fit = meta_multilevel(effect, [.09] * len(effect))
    robust = meta_cluster_robust(fit, [0,0,1,1,2,2,3,3])
    cluster_means = np.asarray(effect).reshape(-1, 2).mean(axis=1)
    expected = cluster_means.std(ddof=1) / np.sqrt(len(cluster_means))
    np.testing.assert_allclose(robust.inference.standard_errors, [expected], atol=5e-15)
    np.testing.assert_allclose(robust.inference.degrees_of_freedom, [3], atol=5e-12)
    np.testing.assert_allclose(robust.omnibus.p_value, robust.inference.p_values[0], atol=5e-15)


def test_invalid_fit_and_robust_inputs():
    y, v, ids = [.1,.4,.8,.5], [.04,.05,.035,.06], [0,0,1,1]
    with pytest.raises(ValueError, match="identifiable"):
        meta_multilevel(y, v, random_intercepts={"one": ids, "two": ids})
    with pytest.raises(ValueError, match="full-rank"):
        meta_multilevel(y, v, moderators=[1] * len(y))
    with pytest.raises(ValueError, match="positive semidefinite"):
        meta_multilevel(y, np.diag([.01,.02,.03,-.01]))
    with pytest.raises(ValueError, match="labels"):
        meta_multilevel(y, v, random_intercepts={"bad": [0,0,1,None]})
    fit = meta_multilevel(y, v, random_intercepts={"cluster": ids}, fixed_variances={"cluster": .1})
    with pytest.raises(ValueError, match="block-diagonal"):
        meta_cluster_robust(fit, [0,1,0,1])
