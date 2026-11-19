import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.risk_scores import score_competing_risks

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "risk_scores"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
COX_CASES = json.loads((FIXTURES / "cox_cases.json").read_text(encoding="utf-8"))
COX_EXPECTED = json.loads((FIXTURES / "cox_expected.json").read_text(encoding="utf-8"))["results"]
FIELDS = ("estimate", "standard_error", "lower", "upper", "influence")


def _compare(actual, target, field):
    np.testing.assert_allclose(
        actual, np.asarray(target, dtype=float), rtol=1e-9, atol=5e-12, err_msg=field,
    )


def _check_case(case, expected):
    arguments = {key: value for key, value in case.items() if key != "id"}
    actual = score_competing_risks(**arguments)
    divergence = case["id"] == "tied-times-and-risks"
    auc_fields = ("estimate",) if divergence else FIELDS
    _compare(actual.horizons, expected["horizons"], "horizons")
    for metric, fields in (("auc", auc_fields), ("brier", FIELDS)):
        summary = getattr(actual, metric)
        for field in fields:
            _compare(getattr(summary, field), [row[field] for row in expected[metric]], field)
    if divergence:
        # The released R reference has a reproducible AUC IF difference for cause 2.
        assert np.max(np.abs(
            actual.auc.standard_error - [row["standard_error"] for row in expected["auc"]]
        )) > 0.2

    for contrast, reference in zip(actual.contrasts, expected["contrasts"], strict=True):
        assert contrast.reference == reference["reference"]
        assert contrast.model == reference["model"]
        for metric, fields in (("auc", auc_fields), ("brier", FIELDS)):
            summary = getattr(contrast, metric)
            for field in fields:
                _compare(getattr(summary, field), reference[metric][field], field)
            if metric == "brier" or not divergence:
                _compare(getattr(contrast, f"{metric}_pvalues"), reference[f"{metric}_pvalues"], metric)


def _check_cox_case(case, expected):
    arguments = {key: value for key, value in case.items() if key != "id"}
    actual = score_competing_risks(**arguments)
    _compare(actual.horizons, expected["horizons"], "horizons")
    for metric in ("auc", "brier"):
        summary = getattr(actual, metric)
        for field in FIELDS:
            _compare(getattr(summary, field), [row[field] for row in expected[metric]], field)
    _compare(actual.integrated_brier, expected["integrated_brier"], "integrated_brier")
    for contrast, reference in zip(actual.contrasts, expected["contrasts"], strict=True):
        assert contrast.reference == reference["reference"]
        assert contrast.model == reference["model"]
        for metric in ("auc", "brier"):
            summary = getattr(contrast, metric)
            for field in FIELDS:
                _compare(getattr(summary, field), reference[metric][field], field)
            _compare(
                getattr(contrast, f"{metric}_pvalues"), reference[f"{metric}_pvalues"], metric,
            )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_matches_recorded_r(case):
    _check_case(case, EXPECTED[case["id"]])


@pytest.mark.parametrize("case", COX_CASES, ids=[case["id"] for case in COX_CASES])
def test_cox_censoring_matches_recorded_r(case):
    _check_cox_case(case, COX_EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    references = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_case(case, references[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_cox_censoring_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cox_cases.json")],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    references = json.loads(process.stdout)["results"]
    for case in COX_CASES:
        _check_cox_case(case, references[case["id"]])


def test_status_labels_and_row_order_do_not_change_the_estimand():
    case = CASES[2]
    arguments = {key: value for key, value in case.items() if key != "id"}
    expected = score_competing_risks(**arguments)
    order = np.random.default_rng(5).permutation(len(case["time"]))
    actual = score_competing_risks(
        np.asarray(case["time"])[order],
        np.choose(np.asarray(case["status"])[order], [0, 9, 7]),
        case["horizons"],
        np.asarray(case["risks"])[:, order, :],
        cause=7,
    )
    for metric in ("auc", "brier"):
        observed, reference = getattr(actual, metric), getattr(expected, metric)
        _compare(observed.estimate, reference.estimate, metric)
        _compare(observed.influence, reference.influence[:, :, order], metric)


def test_undefined_auc_is_nan_and_brier_is_available():
    result = score_competing_risks([2, 3, 4], [0, 2, 1], [1], [[0.1], [0.2], [0.3]])
    assert np.isnan(result.auc.estimate[0, 0])
    assert result.brier.estimate[0, 0] == pytest.approx((0.01 + 0.04 + 0.09) / 3)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"horizons": [4, 1]}, "strictly"),
        ({"status": [0, 1.5, 2]}, "integer codes"),
        ({"risks": [[1.2], [0.1], [0.2]]}, "probability"),
        ({"cause": 0}, "positive integer"),
    ],
)
def test_invalid_inputs(options, message):
    arguments = {"time": [2, 3, 4], "status": [0, 1, 2], "horizons": [2],
                 "risks": [[0.2], [0.1], [0.2]]}
    with pytest.raises(ValueError, match=message):
        score_competing_risks(**(arguments | options))


def test_censoring_inputs_must_describe_the_sample():
    arguments = {"time": [2, 3, 4], "status": [0, 1, 2], "horizons": [2],
                 "risks": [[0.2], [0.1], [0.2]]}
    with pytest.raises(ValueError, match="sample-aligned"):
        score_competing_risks(**arguments, censoring_covariates=[1.0, 2.0])
    with pytest.raises(ValueError, match="nonconstant"):
        score_competing_risks(**arguments, censoring_covariates=[1.0, 1.0, 1.0])
