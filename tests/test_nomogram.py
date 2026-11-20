import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.special import expit

from biomed_parity.calibration import _logistic_fit
from biomed_parity.nomogram import NomogramTerm, nomogram
from biomed_parity.rms_prediction import datadist
from biomed_parity.splines import fit_restricted_cubic_spline


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "nomogram"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _fit(case):
    j = np.arange(180)
    age = ((j * 13 % 41) - 20) / 4
    z = ((j * 7 % 19) - 9) / 7
    treated = j % 2
    data = dict(age=age, arm=np.where(treated == 1, "therapy", "control"), z=z)
    distribution = datadist(data, factor_levels={"arm": ["control", "therapy"]})
    basis = fit_restricted_cubic_spline(age, knots=[-4, -1, 1.5, 4])
    interacting = case["model"] == "lrm"

    def design(grid):
        arm = (grid["arm"] == "therapy").astype(float)
        raw = basis.transform(grid["age"])
        columns = [np.ones(len(arm)), raw, arm, grid["z"]]
        if interacting:
            columns.append(raw * arm[:, None])
        return np.column_stack(columns)

    x = design(data)
    if interacting:
        prob = expit(-0.35 + .16 * age + .08 * age**2 + .3 * z + treated * (.3 - .2 * age))
        y = ((j * 8917 % 487) / 487 < prob).astype(float)
        beta = _logistic_fit(x[:, 1:], y)
        terms = {
            "age": NomogramTerm([1, 2, 3, 4, 6, 7, 8], case["age"], by={"arm": None}),
            "z": NomogramTerm([5], case["z"]),
        }
        outcome = expit
    else:
        y = .75 + .21 * age + .1 * age**2 + .48 * z - .8 * treated + .2 * np.sin(j)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        terms = {
            "age": NomogramTerm([1, 2, 3], case["age"]), "arm": NomogramTerm([4]),
            "z": NomogramTerm([5], case["z"]),
        }
        outcome = np.exp
    result = nomogram(
        distribution, terms, design, beta, max_points=case["max_points"],
        linear_ticks=case["lp"], outcomes={"outcome": (outcome, case["outcome_at"])},
    )
    return result, design, beta


def _compare(case, reference):
    result, design, beta = _fit(case)
    assert len(result.predictor_axes) == len(reference["axes"])
    for native, expected in zip(result.predictor_axes, reference["axes"], strict=True):
        assert native.predictor == expected["predictor"]
        assert native.by.get("arm") == expected["by"]
        np.testing.assert_array_equal(native.ticks, expected["ticks"])
        np.testing.assert_allclose(native.effects, expected["effects"], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(native.points, expected["points"], rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(result.points_per_unit, reference["scale"], rtol=2e-6)
    np.testing.assert_allclose(result.zero_point_predictor, reference["zero"], rtol=1e-6, atol=2e-7)
    np.testing.assert_allclose(result.linear_axis.points, reference["lp_points"], rtol=2e-6, atol=2e-5)
    outcome = result.outcome_axes["outcome"]
    np.testing.assert_allclose(outcome.ticks, reference["outcome_label"], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(outcome.linear_predictor, reference["outcome_lp"], atol=1e-13)
    np.testing.assert_allclose(outcome.points, reference["outcome_points"], rtol=2e-6, atol=2e-5)

    total = 0
    predictors = {"age": np.array([2]), "arm": np.array(["therapy"]), "z": np.array([1])}
    for axis in result.predictor_axes:
        if all(predictors[name][0] == value for name, value in axis.by.items()):
            chosen = axis.ticks == predictors[axis.predictor][0]
            total += axis.points[chosen][0]
    np.testing.assert_allclose(result.to_linear_predictor(total), np.sum(design(predictors)[0] * beta),
                               atol=1e-13, rtol=0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_nomogram_matches_rms_snapshot(case):
    _compare(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_nomogram_matches_live_rms():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    expected = json.loads(run.stdout)["results"]
    for case in CASES:
        _compare(case, expected[case["id"]])


def test_nomogram_rejects_overlapping_terms_and_unknown_numeric_interaction_ticks():
    distribution = datadist({"x": [1, 2, 3, 4], "z": [1, 2, 3, 4]})

    def design(grid):
        return np.column_stack((np.ones(len(grid["x"])), grid["x"], grid["z"]))

    with pytest.raises(ValueError, match="partition"):
        nomogram(distribution, {"x": NomogramTerm([1, 2]), "z": NomogramTerm([2])}, design, [0, 5, 1])
    with pytest.raises(ValueError, match="give interaction levels"):
        nomogram(distribution, {"x": NomogramTerm([1, 2], by={"z": None})}, design, [0, 5, 1])
