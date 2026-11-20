import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.survey_calibration import calibrate, calibrate_replicates, calibrated_total
from biomed_parity.survey_replicates import svrepdesign


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "survey_calibration"
FIXTURE = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
OBS = FIXTURE["observations"]
CASES = FIXTURE["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _assert_fields(result, expected):
    for field, target in expected.items():
        actual = getattr(result, field)
        assert np.shape(actual) == np.shape(target), field
        np.testing.assert_allclose(actual, target, rtol=1e-10, atol=2e-10, err_msg=field)


def _check_r_case(case, reference):
    options = {key: case[key] for key in ("calfun", "variance", "maxit") if key in case}
    args = (OBS["weights"], case["variables"], case["population"])
    reference = reference[case["name"]]
    if reference["converged"]:
        fit = calibrate(*args, epsilon=1e-12, **options)
        assert 0 < fit.iterations <= 50
        np.testing.assert_allclose(fit.control_totals, case["population"], rtol=1e-12)
    else:
        with pytest.warns(RuntimeWarning, match="maximum relative control error"):
            fit = calibrate(*args, epsilon=1e-12, force=True, **options)
        assert fit.iterations == case["maxit"]
    _assert_fields(fit, {key: value for key, value in reference.items() if key not in ("taylor_total", "replication")})

    if "taylor_total" in reference:
        total = calibrated_total(OBS["outcomes"], *args, psu=OBS["psu"], strata=OBS["strata"], fpc=OBS["fpc"],
                                 epsilon=1e-12, **options)
        _assert_fields(total, reference["taylor_total"])
        np.testing.assert_allclose(total.calibration.weights, fit.weights)

    if "replication" in case:
        factors = np.asarray(OBS["repweight_factors"])
        replication = case["replication"]
        if replication["combined_weights"]:
            factors = factors * np.asarray(OBS["weights"])[:, None]
        design = svrepdesign(factors, OBS["weights"], mse=True, **replication)
        weighted = calibrate_replicates(design, case["variables"], case["population"], epsilon=1e-12, **options)
        achieved = np.asarray(case["variables"]).T @ weighted.analysis_weights
        target = np.broadcast_to(np.asarray(case["population"])[:, None], achieved.shape)
        np.testing.assert_allclose(achieved, target, rtol=1e-12)
        _assert_fields(weighted, {"analysis_weights": reference["replication"]["analysis_weights"],
                                  "design_df": reference["replication"]["df"]})
        _assert_fields(weighted.total(OBS["outcomes"]), reference["replication"]["total"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_matches_recorded_r(case):
    _check_r_case(case, EXPECTED)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    results = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_r_case(case, results)


def test_invariant_to_row_order_and_control_units():
    case = CASES[0]
    weights = np.asarray(OBS["weights"])
    controls = np.asarray(case["variables"])
    population = np.asarray(case["population"])
    order = [2, 5, 6, 1, 0, 4, 7, 3]
    scales = [1e-3, 1, 1e6]
    for calfun in ("linear", "raking"):
        ordinary = calibrate(weights, controls, population, calfun=calfun, epsilon=1e-12)
        alternate = calibrate(weights[order], controls[order] * scales, population * scales, calfun=calfun, epsilon=1e-12)
        np.testing.assert_allclose(ordinary.multipliers[order], alternate.multipliers, rtol=5e-10)


def test_zero_base_weight_is_preserved_and_exact_controls_have_zero_taylor_variance():
    weights = [1, 2, 0, 4]
    controls = [[1, 0], [1, 1], [1, 1], [1, 0]]
    for calfun in ("linear", "raking"):
        fit = calibrated_total(controls, weights, controls, [18, 8], calfun=calfun, epsilon=1e-12)
        np.testing.assert_allclose(fit.estimate, [18, 8], rtol=1e-12)
        np.testing.assert_allclose(fit.covariance, 0, atol=1e-25)
        assert fit.calibration.weights[2] == 0


def test_stopping_and_impossible_support():
    with pytest.raises(RuntimeError, match="calibration failed to converge"):
        calibrate([1, 2], [1, 1], [6], calfun="raking", maxit=1)
    with pytest.raises(ValueError, match="inconsistent with the positive-weight sample"):
        calibrate([1, 2], [[1, 1], [1, 1]], [4, 5], calfun="raking")
    with pytest.raises(ValueError, match="requires independent control"):
        calibrate([1, 2], [[1, 1], [1, 1]], [4, 4])
    design = svrepdesign([[0, 2], [2, 0], [2, 0]])
    with pytest.raises(ValueError, match="replicate 1"):
        calibrate_replicates(design, [[1, 1], [1, 0], [1, 0]], [4, 2], calfun="raking")


def test_single_control_accepts_scalar_total_and_variance_coefficient():
    fit = calibrate([1, 2, 3], [1, 1, 1], 12, variance=0.5, epsilon=[1e-10])
    np.testing.assert_allclose(fit.weights, [2, 4, 6])


@pytest.mark.parametrize("additional,message", [
    ({"weights": [-1, 2]}, "nonnegative"),
    ({"weights": [0, 0]}, "at least one positive"),
    ({"variables": [[1, 2]]}, "row per weight"),
    ({"population": [1, 2]}, "one total per control"),
    ({"variance": [0, 1]}, "strictly positive"),
    ({"epsilon": [1e-7, 1e-7]}, "per control"),
    ({"calfun": "logit"}, "linear.*raking"),
    ({"maxit": True}, "positive integer"),
])
def test_invalid_inputs(additional, message):
    args = {"weights": [1, 2], "variables": [1, 1], "population": [6]}
    with pytest.raises(ValueError, match=message):
        calibrate(**(args | additional))
