import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.competing_calibration import _local_risk
from biomed_parity.risk_cross_validation import cross_validate_competing_risks


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "risk_cross_validation"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _check(case, reference):
    j = np.arange(42)
    times, status, markers = j + 1, (j * 17 // 5) % 3, j % 2
    sites = (j // 3) % 2

    def risk(training, horizons):
        return [_local_risk(times[training], status[training], case["cause"], horizon)
                for horizon in horizons]

    def marginal(training, held_out, horizons):
        return np.broadcast_to(risk(training, horizons), (len(held_out), len(horizons)))

    def stratified(labels):
        def fit(training, held_out, horizons):
            estimates = np.array([risk(training[labels[training] == level], horizons) for level in (0, 1)])
            return estimates[labels[held_out]]
        return fit

    models = {"grouped": stratified(markers), "alternative": stratified(sites), "marginal": marginal}

    result = cross_validate_competing_risks(
        times, status, case["horizons"], models,
        folds=reference["folds"], cause=case["cause"],
        censoring_covariates=np.cos(j)[:, None] if case.get("censoring") == "cox" else None,
        auc_inference="riskregression",
    )
    assert result.model_names == ("grouped", "alternative", "marginal")
    np.testing.assert_array_equal(result.folds, reference["folds"])
    np.testing.assert_allclose(result.predictions[:2], reference["predictions"], rtol=0, atol=2e-15)
    np.testing.assert_allclose(result.predictions[2], reference["marginal_predictions"], rtol=0, atol=2e-15)
    for metric in ("auc", "brier"):
        summary = getattr(result.scores, metric)
        for field, expected in reference[metric].items():
            np.testing.assert_allclose(getattr(summary, field)[:2], expected, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(result.scores.integrated_brier[:2], reference["integrated_brier"],
                               atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_riskregression_cv_snapshot(case):
    reference = EXPECTED["results"][case["id"]]
    _check(case, reference)
    assert reference["malformed_marginal_cells"] > 0


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_riskregression_cv():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    expected = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, expected[case["id"]])


def test_automatic_folds_are_disjoint_balanced_and_reproducible():
    held_out_rows = []

    def model(train, test, horizons):
        assert not train.flags.writeable and not test.flags.writeable and not horizons.flags.writeable
        assert np.intersect1d(train, test).size == 0
        held_out_rows.extend(test)
        return np.repeat((test / 12)[:, None], len(horizons), axis=1)

    args = (np.arange(1, 12), np.arange(11) % 3, [5, 7], {"model": model})
    result = cross_validate_competing_risks(*args, n_folds=4, seed=524)
    np.testing.assert_array_equal(np.sort(held_out_rows), np.arange(11))
    assert np.ptp(np.bincount(result.folds)) <= 1
    replay = cross_validate_competing_risks(*args, n_folds=4, seed=524)
    np.testing.assert_array_equal(result.folds, replay.folds)
    np.testing.assert_array_equal(result.predictions, replay.predictions)


def test_invalid_model_shape_identifies_the_fold():
    with pytest.raises(ValueError, match="model 'example', fold 0"):
        cross_validate_competing_risks(
            [1, 2, 3], [1, 0, 2], [2], {"example": lambda train, test, horizons: [0.1]},
            folds=[0, 1, 1],
        )
