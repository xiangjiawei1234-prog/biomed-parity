import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.mstate_covariates import expand_covariates
from biomed_parity.mstate_fit import msfit
from biomed_parity.mstate_prep import msprep


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "mstate_covariates"
SPEC = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _records():
    return msprep(SPEC["time"], SPEC["status"], transitions=[[0, 1], [0, 2], [1, 2]],
                  keep={name: np.asarray(values, dtype=object) for name, values in SPEC["keep"].items()})


def _check(case, reference):
    records = _records()
    options = {key: val for key, val in case.items() if key != "id"}
    options.setdefault("append", False)
    expanded = expand_covariates(records, **options)
    if options["append"]:
        np.testing.assert_array_equal(expanded.transition, records.transition)
        np.testing.assert_array_equal(expanded.t_start, records.t_start)
        assert list(expanded.covariates)[:len(records.covariates)] == list(records.covariates)
        for key, value in records.covariates.items():
            np.testing.assert_array_equal(expanded.covariates[key], value)
        expanded = {key: value for key, value in expanded.covariates.items()
                    if key not in records.covariates}
    np.testing.assert_array_equal(records.transition, np.asarray(reference["trans"]) - 1)
    assert list(expanded) == list(reference["expanded"])
    for term, values in expanded.items():
        np.testing.assert_allclose(values, np.asarray(reference["expanded"][term], dtype=float),
                                   rtol=0, atol=2e-14)


@pytest.mark.parametrize("case", SPEC["cases"], ids=lambda case: case["id"])
def test_matches_recorded_mstate(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_mstate():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, check=True, text=True, timeout=60,
    )
    references = json.loads(run.stdout)["results"]
    for case in SPEC["cases"]:
        _check(case, references[case["id"]])


def test_an_unobserved_declared_edge_has_zeros_and_its_name():
    records = msprep(
        [[None, 1, 1, 10], [None, 2, 2, 7]],
        [[None, 0, 1, 0], [None, 0, 1, 0]],
        transitions=[[0, 2], [0, 1], [1, 3]], keep={"age": [22, 37]},
    )
    terms = expand_covariates(records, "age", append=False)
    np.testing.assert_array_equal(terms["age.3"], 0)
    np.testing.assert_array_equal(terms["age.1"], [0, 22, 0, 37])


def test_covariates_reject_bad_levels_and_name_collisions():
    records = _records()
    with pytest.raises(ValueError, match="undeclared"):
        expand_covariates(records, "stage", factor_levels={"stage": ["early", "late stage"]})
    with pytest.raises(ValueError, match="already exists"):
        expand_covariates(expand_covariates(records, "age"), "age")


def test_transition_specific_coefficients_match_separate_cox_fits():
    subject = np.arange(72)
    followup = 2 + (subject * 5 % 19) / 3
    stop = followup + 1 + (subject * 3 % 23) / 4
    ill = subject % 4 != 2
    dead = (subject * 3 % 11) >= 4
    records = msprep(
        np.column_stack((np.full(72, np.nan), np.where(ill, followup, stop), stop)),
        np.column_stack((np.full(72, np.nan), ill, dead)),
        transitions=[[0, 1], [0, 2], [1, 2]], keep={"marker": np.sin(subject * 1.4)},
    )
    columns = expand_covariates(records, "marker", append=False)
    combined = msfit(records, covariates=np.column_stack(tuple(columns.values())))
    separate = []
    for edge in range(len(records.transitions)):
        rows = records.transition == edge
        fitted = msfit(
            records.t_start[rows], records.t_stop[rows], records.status[rows],
            np.zeros(rows.sum(), dtype=int), [[0, 1]],
            covariates=records.covariates["marker"][rows, None],
        )
        separate.append(fitted.coefficients[0])
    np.testing.assert_allclose(combined.coefficients, separate, rtol=0, atol=2e-8)
