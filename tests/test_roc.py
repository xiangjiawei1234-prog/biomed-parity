"""Compare native DeLong inference with pROC oracle outputs."""

import csv
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.roc import delong_auc, delong_paired_test, delong_unpaired_test


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "roc"


def read_tsv(lines):
    return list(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))


def fixture_rows(name):
    return read_tsv((FIXTURES / name).read_text().splitlines())


def compare_fields(actual, expected):
    for name, value in actual.items():
        target = float(expected[name])
        np.testing.assert_allclose(value, target, rtol=2e-12, atol=2e-14, err_msg=name)


@pytest.mark.parametrize("scenario", fixture_rows("scenarios.tsv"), ids=lambda case: case["id"])
def test_delong_matches_proc(scenario):
    dataset = [
        row for row in fixture_rows("observations.tsv")
        if row["dataset"] == scenario["dataset"]
    ]
    cohort_1 = [row for row in dataset if row["cohort"] == "1"]
    labels_1 = [int(row["label"]) for row in cohort_1]
    scores_1 = [float(row["score_1"]) for row in cohort_1]
    method = scenario["method"]
    direction = scenario["direction"]
    alternative = scenario["alternative"]

    if method == "auc":
        result = delong_auc(
            labels_1, scores_1, direction=direction,
            confidence_level=float(scenario["confidence_level"]),
        )
        actual = {
            "auc_1": result.auc,
            "variance_1": result.variance,
            "lower": result.confidence_interval[0],
            "upper": result.confidence_interval[1],
        }
    elif method == "paired":
        scores_2 = [float(row["score_2"]) for row in cohort_1]
        result = delong_paired_test(
            labels_1, scores_1, scores_2, direction=direction, alternative=alternative,
            confidence_level=float(scenario["confidence_level"]),
        )
        actual = vars(result).copy()
        actual["lower"], actual["upper"] = actual.pop("confidence_interval")
    else:
        cohort_2 = [row for row in dataset if row["cohort"] == "2"]
        labels_2 = [int(row["label"]) for row in cohort_2]
        scores_2 = [float(row["score_1"]) for row in cohort_2]
        result = delong_unpaired_test(
            labels_1, scores_1, labels_2, scores_2, direction=direction, alternative=alternative
        )
        actual = vars(result)

    expected = next(row for row in fixture_rows("expected.tsv") if row["id"] == scenario["id"])
    compare_fields(actual, expected)


@pytest.mark.skipif(
    os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 to run pROC"
)
def test_regenerate_roc_oracle():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    output = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "generate.R"), str(FIXTURES)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    live = read_tsv(output.stdout.splitlines())
    expected = fixture_rows("expected.tsv")
    assert len(live) == len(expected)
    for actual, reference in zip(live, expected, strict=True):
        assert actual["id"] == reference["id"]
        assert actual.keys() == reference.keys()
        assert {field for field, value in actual.items() if value == "NA"} == {
            field for field, value in reference.items() if value == "NA"
        }
        compare_fields(
            {field: float(value) for field, value in actual.items() if field != "id" and value != "NA"},
            reference,
        )


@pytest.mark.parametrize(
    ("labels", "scores", "match"),
    [
        ([0, 0, 1], [0, 1, 2], "at least two"),
        ([0, 1, 1, 1], [0, 1, 2, 3], "at least two"),
        ([0, 0, 0, 0], [0, 1, 2, 3], "at least two"),
        ([0, 0, 1, 2], [0, 1, 2, 3], "labels must be 0"),
        ([0, 0, 1, np.nan], [0, 1, 2, 3], "labels must be 0"),
        ([0, 0, 1, 1], [0, 1, 2, np.inf], "finite real numbers"),
        ([0, 0, 1, 1], [0, 1, 2, np.nan], "finite real numbers"),
        ([0, 0, 1, 1], [0, 1, 2, 1j], "finite real numbers"),
        ([0, 0, 1, 1], [[0, 1, 2, 3]], "one-dimensional"),
        ([0, 0, 1, 1], [0, 1, 2], "equal size"),
    ],
)
def test_delong_rejects_invalid_observations(labels, scores, match):
    with pytest.raises(ValueError, match=match):
        delong_auc(labels, scores)


@pytest.mark.parametrize("level", [-0.1, 1.1, np.nan])
def test_delong_rejects_invalid_confidence_level(level):
    with pytest.raises(ValueError, match="confidence_level"):
        delong_auc([0, 0, 1, 1], [0, 1, 2, 3], confidence_level=level)


def test_delong_rejects_implicit_direction_and_unknown_alternative():
    labels = [0, 0, 1, 1]
    scores = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="direction"):
        delong_auc(labels, scores, direction="auto")
    with pytest.raises(ValueError, match="alternative"):
        delong_paired_test(labels, scores, scores, alternative="one-sided")


@pytest.mark.parametrize("direction", ["<", ">"])
def test_delong_matches_pairwise_kernel(direction):
    rng = np.random.default_rng(185)
    labels = rng.permutation([0] * 7 + [1] * 29)
    scores_1, scores_2 = rng.integers(-5, 6, size=(2, len(labels)))

    def kernel(scores):
        cases, controls = scores[labels == 1, None], scores[labels == 0][None, :]
        wins = cases > controls if direction == "<" else cases < controls
        return wins + 0.5 * (cases == controls)

    def covariance(first, second):
        return sum(
            np.cov(first.mean(axis=axis), second.mean(axis=axis))[0, 1] / first.shape[1 - axis]
            for axis in (0, 1)
        )

    first, second = kernel(scores_1), kernel(scores_2)
    result = delong_paired_test(labels, scores_1, scores_2, direction=direction)
    np.testing.assert_allclose(
        (result.auc_1, result.auc_2, result.variance_1, result.variance_2, result.covariance),
        (first.mean(), second.mean(), covariance(first, first), covariance(second, second),
         covariance(first, second)),
        atol=2e-14,
    )
