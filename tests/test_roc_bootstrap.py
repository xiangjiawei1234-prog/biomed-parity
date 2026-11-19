"""Compare binary ROC areas and fixed-draw bootstrap inference with pROC."""

import csv
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_bootstrap import bootstrap_auc_ci, bootstrap_paired_test, roc_auc


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "roc_bootstrap"


def read_tsv(lines):
    return list(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))


def fixture(name):
    return read_tsv((FIXTURES / f"{name}.tsv").read_text().splitlines())


OBSERVATIONS = fixture("observations")
LABELS = [int(row["label"]) for row in OBSERVATIONS]
SCENARIOS = fixture("scenarios")
EXPECTED = {row["id"]: row for row in fixture("expected")}
DRAWS = np.loadtxt(FIXTURES / "draws.tsv", delimiter="\t", skiprows=1, dtype=int)


def scores(name):
    return [float(row[name]) for row in OBSERVATIONS]


def assert_matches(actual, reference):
    assert actual.keys() == {field for field, value in reference.items() if field != "id" and value != "NA"}
    for field, value in actual.items():
        np.testing.assert_allclose(value, float(reference[field]), rtol=2e-12, atol=2e-14, err_msg=field)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda row: row["id"])
def test_roc_bootstrap_matches_proc(scenario):
    settings = {
        "direction": scenario["direction"],
        "partial_auc": (
            None if scenario["low"] == "NA" else (float(scenario["low"]), float(scenario["high"]))
        ),
        "focus": scenario["focus"],
        "standardize": scenario["standardize"] == "TRUE",
        "allow_below_chance": scenario["allow_below_chance"] == "TRUE",
        "percent": scenario["percent"] == "TRUE",
    }
    first = scores(scenario["first"])
    method = scenario["method"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="partial AUC correction")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="nonfinite bootstrap")
        if method == "auc":
            actual = {"auc_1": roc_auc(LABELS, first, **settings)}
        elif method == "ci":
            ci = bootstrap_auc_ci(
                LABELS, first, **settings, bootstrap_indices=DRAWS,
                confidence_level=float(scenario["confidence_level"]),
            )
            actual = {
                "auc_1": ci.auc, "median": ci.median, "lower": ci.confidence_interval[0],
                "upper": ci.confidence_interval[1], "n_resamples": ci.n_resamples,
            }
        else:
            actual = asdict(bootstrap_paired_test(
                LABELS, first, scores(scenario["second"]), **settings,
                bootstrap_indices=DRAWS, alternative=scenario["alternative"],
            ))
    assert_matches(actual, EXPECTED[scenario["id"]])


@pytest.mark.skipif(
    os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 to run pROC"
)
def test_regenerate_roc_bootstrap_oracle():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    command = ["Rscript", "--vanilla", str(FIXTURES / "generate.R"), str(FIXTURES)]
    draw_text = subprocess.run(
        [*command, "--draws"], cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    ).stdout
    np.testing.assert_array_equal(np.loadtxt(draw_text.splitlines()[1:], dtype=int), DRAWS)
    expected_text = subprocess.run(
        command, cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    ).stdout
    live = {row["id"]: row for row in read_tsv(expected_text.splitlines())}
    assert live.keys() == EXPECTED.keys()
    for case_id, result in live.items():
        values = {field: float(value) for field, value in result.items() if field != "id" and value != "NA"}
        assert_matches(values, EXPECTED[case_id])


def test_roc_auc_with_one_case_and_control():
    assert roc_auc([0, 1], [1, 2]) == 1
    assert roc_auc([0, 1], [1, 1]) == 0.5
    assert roc_auc([0, 1], [2, 1]) == 0


@pytest.mark.parametrize("focus", ["specificity", "sensitivity"])
def test_partial_areas_of_ideal_and_diagonal_curves(focus):
    labels = [0, 1, 1, 0, 1]
    kwargs = {"partial_auc": (0.15, 0.6), "focus": focus}
    assert roc_auc(labels, labels, **kwargs) == pytest.approx(0.45)
    assert roc_auc(labels, [1] * 5, **kwargs) == pytest.approx(0.45 * (1 - 0.375))
    assert roc_auc(labels, labels, **kwargs, standardize=True) == pytest.approx(1)
    assert roc_auc(labels, [1] * 5, **kwargs, standardize=True) == pytest.approx(0.5)


def test_standardized_below_chance_warning_and_bootstrap_keeps_draws():
    with pytest.warns(RuntimeWarning, match="below the chance diagonal"):
        result = bootstrap_auc_ci(
            LABELS, scores("score_poor"), partial_auc=(0.8, 1), standardize=True,
            bootstrap_indices=DRAWS,
        )
    assert np.isnan(result.auc)
    assert result.confidence_interval[0] < 0.5
    assert result.n_resamples == DRAWS.shape[0]


def test_seeded_bootstrap_is_repeatable_and_identical_score_test_is_null():
    first = scores("score_1")
    ci = bootstrap_auc_ci(LABELS, first, random_state=312, n_resamples=64)
    assert ci == bootstrap_auc_ci(LABELS, first, random_state=312, n_resamples=64)
    comparison = bootstrap_paired_test(LABELS, first, first, random_state=312, n_resamples=64)
    assert comparison.statistic == 0
    assert comparison.p_value == 1
    assert comparison.n_resamples == 64


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.zeros((1, 7), dtype=int), "at least two"),
        (np.zeros((2, 7)), "zero-based integer"),
        (np.full((2, 7), 7), "zero-based integer"),
        (np.zeros((2, 7), dtype=int), "class counts"),
    ],
)
def test_bootstrap_indices_must_be_in_range_and_stratified(indices, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_auc_ci(LABELS, scores("score_1"), bootstrap_indices=indices)


@pytest.mark.parametrize("limits", [(-0.5, 0.7), (0.6, 1.1), (0.1,), (np.nan, 0.7)])
def test_partial_bounds_require_two_finite_unit_values(limits):
    with pytest.raises(ValueError, match="partial_auc"):
        roc_auc(LABELS, scores("score_1"), partial_auc=limits)


def test_roc_rejects_single_class():
    with pytest.raises(ValueError, match="case and one control"):
        roc_auc([0, 0], [1, 2])
