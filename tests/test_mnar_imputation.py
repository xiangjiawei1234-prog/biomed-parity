import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.chained_imputation import mice_impute


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "mnar_imputation"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _draws(case, offset):
    data = np.column_stack((case["x"], case["z"], np.asarray(case["y"], dtype=float)))
    method = case["method"] if offset is not None else case["method"].split(".")[1]
    result = mice_impute(
        data, methods=[None, None, method],
        mnar_offsets=None if offset is None else {2: offset}, m=1000, iterations=1, seed=1913,
    )
    np.testing.assert_array_equal(result.completed[:, ~result.where[:, 2], 2],
                                  np.broadcast_to(data[~result.where[:, 2], 2], (1000, (~result.where[:, 2]).sum())))
    return result.completed[:, result.where[:, 2], 2]


def _check(case, reference):
    offset = case["delta"] + case["x_weight"] * np.asarray(case["x"]) + case["aux_weight"] * np.asarray(case["aux"])
    draws = _draws(case, offset)
    base = _draws(case, None)
    targets = offset[~np.isfinite(np.asarray(case["y"], dtype=float))]
    if case["method"] == "mnar.norm":
        np.testing.assert_allclose(draws - base, np.broadcast_to(targets, draws.shape), atol=5e-15, rtol=0)
        assert reference["paired_error"] < 5e-14
    else:
        assert ((draws - base) * targets >= 0).all()
        assert reference["paired_error"] == 0
    se = np.sqrt(draws.var(axis=0, ddof=1) / len(draws) +
                 np.asarray(reference["standard_deviations"])**2 / case["n_draws"])
    assert (abs(draws.mean(axis=0) - reference["means"]) <= 5 * se).all()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_mnar_against_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_mnar_against_live_r():
    captured = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, timeout=60, check=True,
    )
    actual = json.loads(captured.stdout)["results"]
    for case in CASES:
        _check(case, actual[case["id"]])


def test_callable_offset_uses_current_imputations_and_respects_where():
    data = [[1, 11], [2, 12], [np.nan, np.nan], [4, 13], [5, 14], [6, np.nan]]
    where = np.isnan(data)
    where[1, 1] = True
    settings = dict(
        methods=[lambda current: np.full(len(current), 3), "mnar.norm"],
        predictor_matrix=[[0, 0], [1, 0]], m=4, iterations=3, seed=91, where=where,
    )

    def offset(current):
        assert not current.flags.writeable
        return current[:, 0] / 4

    actual = mice_impute(data, mnar_offsets={1: offset}, **settings)
    expect = mice_impute(data, mnar_offsets={1: np.arange(1, 7) / 4}, **settings)
    np.testing.assert_array_equal(actual.completed, expect.completed)
    assert np.all(actual.completed[:, 1, 1] != 12)


@pytest.mark.parametrize("offsets,match", [
    (None, "must specify each"),
    ({0: 1}, "must specify each"),
    ({1: np.inf}, "finite scalar"),
    ({1: [1, 2]}, "finite scalar"),
    ({1: lambda data: np.full(len(data), np.nan)}, "offset callable returned missing"),
])
def test_invalid_mnar_offsets(offsets, match):
    with pytest.raises(ValueError, match=match):
        mice_impute([[1, 2], [2, 3], [3, np.nan]], methods=[None, "mnar.norm"], mnar_offsets=offsets)
