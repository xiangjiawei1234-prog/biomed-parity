import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.competing_calibration import _local_risk
from biomed_parity.risk_bootstrap import bootstrap_validate_competing_risks


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "risk_bootstrap"
CASES = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _check(case, reference):
    j = np.arange(42)
    time, status = j + 1, (j * 17 // 5) % 3

    def stratified(labels):
        def fit(training, held_out, horizons):
            assert not training.flags.writeable and not held_out.flags.writeable and not horizons.flags.writeable
            estimates = []
            for level in (0, 1):
                rows = training[labels[training] == level]
                estimates.append([_local_risk(time[rows], status[rows], case["cause"], horizon)
                                  for horizon in horizons])
            return np.asarray(estimates)[labels[held_out]]
        return fit

    models = {"grouped": stratified(j % 2), "alternative": stratified((j // 3) % 2)}
    result = bootstrap_validate_competing_risks(
        time, status, case["horizons"], models, train_indices=reference["train"], cause=case["cause"],
        censoring_covariates=np.cos(j) if case.get("cox") else None, keep_predictions=True,
    )
    assert result.model_names == tuple(models)
    predictions = np.asarray([
        [row[name] for name in models] for row in reference["predictions"]
    ], dtype=float)
    np.testing.assert_allclose(result.predictions, predictions, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(result.out_of_bag, np.isfinite(predictions[:, 0, :, 0]))
    for metric in ("auc", "brier"):
        summary = getattr(result, metric)
        contrast = getattr(result.contrasts[0], metric)
        for field, values in reference[metric].items():
            np.testing.assert_allclose(getattr(summary, field), np.asarray(values, dtype=float),
                                       atol=2e-8, rtol=2e-7)
        np.testing.assert_allclose(summary.replicates, np.asarray(reference[metric + "_replicates"], dtype=float),
                                   atol=2e-8, rtol=2e-7)
        for field, values in reference[metric + "_contrast"].items():
            np.testing.assert_allclose(getattr(contrast, field), np.asarray(values, dtype=float),
                                       atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(result.integrated_brier.replicates, reference["integrated_brier"],
                               atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_competing_risk_bootstrap_matches_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_competing_risk_bootstrap_matches_live_r():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    result = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, result[case["id"]])


def test_undefined_auc_and_an_empty_held_out_draw_are_retained_as_nan():
    def model(train, validation, horizons):
        np.testing.assert_array_equal(train, [0, 0, 1])
        return np.zeros((len(validation), len(horizons)), dtype=bool)

    result = bootstrap_validate_competing_risks(
        [1, 2, 3], [1, 2, 0], [2], {"example": model}, train_indices=[[0, 1, 2], [0, 0, 1]],
    )
    np.testing.assert_array_equal(result.out_of_bag, [[False, False, False], [False, False, True]])
    assert np.isnan(result.auc.estimate[0, 0])
    assert np.isnan(result.brier.replicates[0, 0, 0])
    assert np.isnan(result.brier.standard_error[0, 0])
    assert result.brier.estimate[0, 0] == 0


def test_generated_subsampling_is_reproducible():
    args = ([1, 2, 3, 4], [1, 0, 2, 2], [2, 3], {
        "marker": lambda train, validation, horizons: np.full((len(validation), len(horizons)), 0.25),
    })
    one = bootstrap_validate_competing_risks(*args, seed=389, n_resamples=7,
                                            training_size=2, replace=False)
    two = bootstrap_validate_competing_risks(*args, seed=389, n_resamples=7,
                                            training_size=2, replace=False)
    np.testing.assert_array_equal(one.training_indices, two.training_indices)
    np.testing.assert_array_equal(one.out_of_bag.sum(axis=1), 2)
    np.testing.assert_array_equal(one.brier.replicates, two.brier.replicates)
