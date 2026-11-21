import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.competing_calibration import _local_risk
from biomed_parity.risk_bootstrap import bootstrap_validate_competing_risks
from biomed_parity.risk_cross_validation import cross_validate_competing_risks
from biomed_parity.risk_resampled import score_resampled_competing_risks


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "risk_resampled"
CASES = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _check(case, reference):
    j = np.arange(42)
    time, status = j + 1, (j * 17 // 5) % 3
    cause = case["cause"]

    def stratified(labels):
        def fit(training, held_out, horizons):
            predictions = []
            for level in (0, 1):
                rows = training[labels[training] == level]
                predictions.append([
                    _local_risk(time[rows], status[rows], cause, horizon) for horizon in horizons
                ])
            return np.asarray(predictions)[labels[held_out]]
        return fit

    models = {"grouped": stratified(j % 2), "alternative": stratified((j // 3) % 2)}
    options = {"cause": cause, "censoring_covariates": np.cos(j) if case.get("cox") else None}
    if case["split"] == "loob":
        predictions = bootstrap_validate_competing_risks(
            time, status, case["horizons"], models,
            train_indices=reference["index"], keep_predictions=True, **options,
        ).predictions
    else:
        predictions = np.stack([
            cross_validate_competing_risks(
                time, status, case["horizons"], models, folds=folds, conservative=True, **options,
            ).predictions
            for folds in reference["index"]
        ])
    reference_predictions = np.asarray(
        [[row[name] for name in models] for row in reference["predictions"]], dtype=float,
    )
    np.testing.assert_allclose(predictions, reference_predictions, rtol=0, atol=2e-15)
    fit = score_resampled_competing_risks(
        time, status, case["horizons"], reference_predictions, auc_method="riskregression", **options,
    )
    for name in ("auc", "brier", "integrated_brier"):
        np.testing.assert_allclose(getattr(fit, name), np.asarray(reference[name], dtype=float),
                                   rtol=2e-7, atol=2e-8, err_msg=name)
    if case["split"] == "loob":
        assert np.any(fit.unobserved_pairs > 0)
    else:
        assert not np.any(fit.unobserved_pairs)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_subject_and_pair_averages_match_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_subject_and_pair_averages_match_live_r():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    results = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, results[case["id"]])


def test_missing_pairs_are_counted_and_left_in_the_auc_denominator():
    # Subjects 0 and 2 are never held out together. Subject 3 is never held out.
    predictions = np.array([[0.8, 0.2, np.nan, np.nan], [np.nan, 0.5, 0.1, np.nan]])
    result = score_resampled_competing_risks(
        [1, 2, 3, 4], [1, 2, 2, 2], [1], predictions[:, :, None],
    )
    np.testing.assert_array_equal(result.subject_validation_counts, [1, 2, 1, 0])
    np.testing.assert_array_equal(result.unobserved_pairs, [1])
    np.testing.assert_allclose(result.auc, [[0.5]])
    np.testing.assert_allclose(result.brier, [[((0.8 - 1)**2 + (0.2**2 + 0.5**2) / 2 + 0.1**2) / 4]])


def test_partial_validation_rows_are_rejected_for_paired_comparisons():
    risks = np.array([[[[0.7], [0.4]], [[np.nan], [0.5]]]])
    with pytest.raises(ValueError, match="shared validation rows"):
        score_resampled_competing_risks([1, 2], [1, 2], [1], risks)
