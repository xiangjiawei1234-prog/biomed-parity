"""Validate a Laplace zero-inflated Poisson mixed model against glmmTMB."""

import csv
from math import exp, factorial, log
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.zero_inflated import (
    fit_zip_random_intercept,
    zip_random_intercept_loglikelihood,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "zero_inflated"


def tsv_rows(lines):
    return list(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))


EXPECTATIONS = tsv_rows((FIXTURES / "expected.tsv").read_text().splitlines())
OBSERVATIONS = tsv_rows((FIXTURES / "observations.tsv").read_text().splitlines())


def observations(case):
    selected = [row for row in OBSERVATIONS if row["dataset"] == case]
    counts = np.array([int(row["count"]) for row in selected])
    group = np.array([row["group"] for row in selected])
    if case == "intercept":
        return counts, group, {}, [0.70], [-0.90]
    ones = np.ones(counts.size)
    count_design = np.column_stack((ones, [int(row["x"]) for row in selected]))
    inflation_design = np.column_stack((ones, [int(row["z"]) for row in selected]))
    offset = [float(row["offset"]) for row in selected]
    design = {"count_design": count_design, "inflation_design": inflation_design, "offset": offset}
    return counts, group, design, [0.68, 0.4], [-0.85, 0.65]


@pytest.mark.parametrize("reference", EXPECTATIONS, ids=lambda row: row["id"])
def test_zip_random_intercept_matches_glmmtmb(reference):
    counts, groups, design, beta, gamma = observations(reference["id"])
    fitted = fit_zip_random_intercept(counts, groups, **design)
    assert fitted.converged, fitted.message
    expected_count = [float(reference["count_intercept"])]
    expected_inflation = [float(reference["zi_intercept"])]
    if reference["id"] != "intercept":
        expected_count.append(float(reference["count_slope"]))
        expected_inflation.append(float(reference["zi_slope"]))
    np.testing.assert_allclose(fitted.count_coefficients, expected_count, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(fitted.inflation_coefficients, expected_inflation, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(fitted.random_intercept_sd, float(reference["sd"]), rtol=2e-6)
    np.testing.assert_allclose(fitted.log_likelihood, float(reference["log_likelihood"]), rtol=0, atol=4e-8)
    evaluated = zip_random_intercept_loglikelihood(counts, groups, beta, gamma, 0.55, **design)
    np.testing.assert_allclose(evaluated, float(reference["trial_log_likelihood"]), rtol=0, atol=4e-8)


@pytest.mark.skipif(
    os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 to run glmmTMB"
)
def test_regenerate_glmmtmb_oracle():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "generate.R"), str(FIXTURES)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    live = tsv_rows(result.stdout.splitlines())
    assert len(live) == len(EXPECTATIONS)
    for actual, expected in zip(live, EXPECTATIONS, strict=True):
        assert actual["id"] == expected["id"]
        assert actual.keys() == expected.keys()
        for field in actual.keys() - {"id"}:
            if expected[field] == "NA":
                assert actual[field] == "NA"
            else:
                np.testing.assert_allclose(
                    float(actual[field]), float(expected[field]), rtol=2e-6, atol=1e-7, err_msg=field,
                )


def test_evaluator_matches_independent_zip_when_sd_is_zero():
    counts = [0, 1, 0, 4, 2]
    means = [1.0, 2.0, 2.5, 3.0, 0.5]
    pi = 0.3
    probabilities = [
        (1 - pi) * exp(-mean) * mean**count / factorial(count) + pi * (count == 0)
        for mean, count in zip(means, counts, strict=True)
    ]
    result = zip_random_intercept_loglikelihood(
        counts, ["a", "a", "b", "a", "b"],
        [0.0], [log(pi / (1 - pi))], 0, offset=np.log(means),
    )
    np.testing.assert_allclose(result, sum(map(log, probabilities)), rtol=0, atol=2e-14)


def test_evaluator_is_invariant_to_row_and_group_order():
    counts, groups, design, beta, gamma = observations("sparse_groups")
    order = np.random.default_rng(2).permutation(len(counts))
    before = zip_random_intercept_loglikelihood(counts, groups, beta, gamma, 0.55, **design)
    shuffled_design = {name: np.asarray(value)[order] for name, value in design.items()}
    after = zip_random_intercept_loglikelihood(
        counts[order], groups[order], beta, gamma, 0.55, **shuffled_design,
    )
    np.testing.assert_allclose(before, after, rtol=0, atol=2e-12)


@pytest.mark.parametrize(
    ("counts", "groups", "options", "message"),
    [
        ([0, 2], ["a"], {}, "equal-length"),
        ([0, -1], ["a", "b"], {}, "nonnegative integers"),
        ([0, 1.5], ["a", "b"], {}, "nonnegative integers"),
        ([0, 1], [0, np.nan], {}, "nonmissing label"),
        ([0, 1], ["a", None], {}, "strings or finite"),
        ([0, 1], ["a", "b"], {"offset": [0, np.inf]}, "offset"),
        ([0, 1], ["a", "b"], {"count_design": [0, 1]}, "two-dimensional"),
        ([0, 1], ["a", "b"], {"inflation_design": np.ones((2, 2))}, "full column rank"),
        ([0, 1], ["a", "a"], {}, "at least two groups"),
        ([1, 1], ["a", "b"], {}, "requires zeros"),
        ([0, 1], ["a", "b"], {"max_iterations": 0}, "positive integer"),
    ],
)
def test_fit_rejects_unsupported_data(counts, groups, options, message):
    with pytest.raises(ValueError, match=message):
        fit_zip_random_intercept(counts, groups, **options)


def test_evaluator_rejects_negative_sd():
    with pytest.raises(ValueError, match="nonnegative"):
        zip_random_intercept_loglikelihood([0, 2], [1, 2], [0.5], [-0.5], -0.1)
