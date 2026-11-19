import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.calibration import calibrate_logistic

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "calibrate_logistic"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["cases"]


def _assert_case(case, reference):
    assert reference["id"] == case["id"]
    kwargs = {key: value for key, value in case.items() if key not in ("id", "B", "r_seed")}
    if "fold_order" in reference:
        kwargs.update(n_resamples=case["B"], fold_order=reference["fold_order"])
    elif "permutation_indices" in reference:
        kwargs["permutation_indices"] = reference["permutation_indices"]
    else:
        kwargs["bootstrap_indices"] = reference["bootstrap_indices"]
    result = calibrate_logistic(**kwargs)
    assert result.failed_replicates == ()
    np.testing.assert_array_equal(result.counts, reference["counts"])
    for field in (
        "predy", "fitted_probabilities", "apparent", "corrected",
        "training_error", "test_error", "optimism", "lower", "upper",
    ):
        actual, expected = getattr(result, field), reference[field]
        if expected is None:
            assert actual is None
            continue
        numbers = [np.nan if value in ("NA", "NaN", None) else value for value in expected]
        np.testing.assert_allclose(actual, numbers, rtol=1e-8, atol=1e-9, err_msg=field)


@pytest.mark.parametrize(
    ("case", "expected"), list(zip(CASES, EXPECTED, strict=True)), ids=[c["id"] for c in CASES],
)
def test_calibration_reproduces_r(case, expected):
    _assert_case(case, expected)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        text=True, check=True, capture_output=True, timeout=60,
    )
    cases = json.loads(process.stdout)["cases"]
    for case, reference in zip(CASES, cases, strict=True):
        _assert_case(case, reference)


def test_custom_smoother_failed_draw_and_grid_support():
    x, y = np.asarray(CASES[0]["x"]), np.asarray(CASES[0]["y"])
    rows = np.arange(len(y))
    failed = np.full(len(y), np.flatnonzero(y == 0)[0])
    calls = []

    def constant_by_sample(probabilities, labels):
        calls.append(len(labels))
        level = np.mean(labels)
        return [0.2, 0.8], [level, level]

    result = calibrate_logistic(
        x, y, predy=[0.0, 0.3, 0.6, 1.0],
        bootstrap_indices=([rows, failed]), smoother=constant_by_sample,
    )
    assert calls == [len(y)] * 3
    assert result.failed_replicates == (1,)
    np.testing.assert_array_equal(result.counts, [0, 1, 1, 0])
    np.testing.assert_allclose(result.apparent, [np.nan, y.mean(), y.mean(), np.nan])
    np.testing.assert_allclose(result.corrected, result.apparent)
    assert np.isnan(result.lower).all()
    assert np.isnan(result.upper).all()


def test_632_requires_observations_in_test_samples():
    x, y = CASES[0]["x"], CASES[0]["y"]
    indices = [np.arange(len(y))]
    with pytest.raises(ValueError, match="omitted at least once"):
        calibrate_logistic(x, y, method=".632", bootstrap_indices=indices)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"predy": [0.5, 1.1]}, "probabilities"),
        ({"fold_order": [1, 2]}, "only to crossvalidation"),
        ({"method": "crossvalidation", "n_resamples": 4, "fold_order": [0, 1]}, "permutation"),
        ({"permutation_indices": [[0, 1]]}, "only to randomization"),
        ({"span": 0}, "span"),
        ({"iterations": -1}, "iterations"),
        ({"delta": np.inf}, "delta"),
        ({"digits": 2.5}, "digits"),
    ],
)
def test_invalid_arguments(options, message):
    with pytest.raises(ValueError, match=message):
        calibrate_logistic(CASES[0]["x"], CASES[0]["y"], **options)
