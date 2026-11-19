"""Validate intercept-only REML and profile intervals against metafor."""

from collections import defaultdict
import csv
from dataclasses import asdict
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.meta_analysis import meta_reml


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "meta_analysis"


def read_tsv(lines):
    return list(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))


def fixture_rows(name):
    return read_tsv((FIXTURES / name).read_text().splitlines())


OBSERVATIONS = defaultdict(list)
for row in fixture_rows("observations.tsv"):
    OBSERVATIONS[row["dataset"]].append((float(row["effect"]), float(row["variance"])))
EXPECTED = {row["id"]: row for row in fixture_rows("expected.tsv")}


@pytest.mark.parametrize("scenario", fixture_rows("scenarios.tsv"), ids=lambda case: case["id"])
def test_meta_reml_matches_metafor(scenario):
    effects, variances = zip(*OBSERVATIONS[scenario["dataset"]], strict=True)
    upper_limit = scenario["tau2_upper_limit"]
    result = meta_reml(
        effects,
        variances,
        test=scenario["test"],
        prediction=scenario["prediction"],
        confidence_level=float(scenario["confidence_level"]),
        tau2_upper_limit=None if upper_limit == "NA" else float(upper_limit),
    )
    actual = asdict(result)
    actual["lower"], actual["upper"] = actual.pop("confidence_interval")
    for label, field in (
        ("prediction", "prediction_interval"),
        ("tau2", "tau2_confidence_interval"),
        ("i2", "i2_confidence_interval"),
        ("h2", "h2_confidence_interval"),
    ):
        actual[f"{label}_lower"], actual[f"{label}_upper"] = actual.pop(field)
    expected = EXPECTED[scenario["id"]]
    assert set(actual) == set(expected) - {"id"}
    for field, value in actual.items():
        np.testing.assert_allclose(value, float(expected[field]), rtol=4e-9, atol=8e-12, err_msg=field)


@pytest.mark.skipif(
    os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 to run metafor"
)
def test_regenerate_metafor_oracle():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "generate.R"), str(FIXTURES)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    live = {row["id"]: row for row in read_tsv(result.stdout.splitlines())}
    assert live.keys() == EXPECTED.keys()
    for case_id, actual in live.items():
        expected = EXPECTED[case_id]
        assert actual.keys() == expected.keys()
        for field in actual.keys() - {"id"}:
            np.testing.assert_allclose(
                float(actual[field]), float(expected[field]),
                rtol=4e-9, atol=8e-12, err_msg=f"{case_id} {field}",
            )


def test_equal_variance_solution_matches_closed_form():
    effects = np.array([0.09, 0.72, 0.5, -0.20])
    result = meta_reml(effects, [0.02] * 4, test="knha")
    np.testing.assert_allclose(result.tau2, effects.var(ddof=1) - 0.02, rtol=2e-13)
    np.testing.assert_allclose(result.mean_effect, effects.mean(), rtol=2e-13)


def test_meta_reml_multiplicative_scale_preservation():
    effects, variances = np.array(OBSERVATIONS["heterogeneous"]).T
    result = meta_reml(effects, variances, test="knha")
    factor = 1e-8
    scaled = meta_reml(effects * factor, variances * factor**2, test="knha")
    np.testing.assert_allclose(
        scaled.confidence_interval, np.asarray(result.confidence_interval) * factor,
        rtol=3e-13, atol=0,
    )
    np.testing.assert_allclose(
        scaled.tau2_confidence_interval, np.asarray(result.tau2_confidence_interval) * factor**2,
        rtol=3e-13, atol=0,
    )


@pytest.mark.parametrize(
    ("effects", "variances", "message"),
    [
        ([0.1], [0.2], "at least two"),
        ([0.1, 0.2], [0.1], "equal-length"),
        ([0.1, 0.2], [[0.1, 0.2]], "equal-length"),
        ([0.1, 0.2], [0.1, 0.0], "strictly positive"),
        ([0.1, 0.2], [0.1, -0.1], "strictly positive"),
        ([0.1, 0.2], [0.1, np.nan], "finite"),
        ([np.nan, 0.2], [0.1, 0.1], "finite"),
        ([0.1j, 0.2], [0.1, 0.1], "real numbers"),
    ],
)
def test_meta_reml_rejects_invalid_data(effects, variances, message):
    with pytest.raises(ValueError, match=message):
        meta_reml(effects, variances)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"test": "f"}, "test"),
        ({"prediction": "bootstrap"}, "prediction"),
        ({"confidence_level": 1}, "confidence_level"),
        ({"confidence_level": 0}, "confidence_level"),
        ({"tau2_upper_limit": -1}, "tau2_upper_limit"),
    ],
)
def test_meta_reml_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        meta_reml([0.1, 0.2], [0.1, 0.1], **options)
