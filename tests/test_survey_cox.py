import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.survey_cox import svycoxph

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "survey_cox"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _check_case(case, expected):
    actual = asdict(svycoxph(**{key: value for key, value in case.items() if key != "id"}))
    assert actual.keys() == expected.keys()
    for field, value in actual.items():
        reference = np.asarray(expected[field], dtype=float)
        assert np.shape(value) == reference.shape
        np.testing.assert_allclose(
            value, reference, rtol=1e-7, atol=1e-9, err_msg=f"{case['id']}.{field}",
        )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_matches_recorded_r(case):
    _check_case(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    results = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_case(case, results[case["id"]])


def test_row_order_and_weight_scale_invariance():
    case = CASES[0]
    arguments = {key: value for key, value in case.items() if key != "id"}
    permutation = np.random.default_rng(9207).permutation(len(case["time"]))
    relabeled = {key: np.asarray(value)[permutation] for key, value in arguments.items()}
    relabeled["weights"] *= 11
    original = asdict(svycoxph(**arguments))
    permuted = asdict(svycoxph(**relabeled))
    for field, value in original.items():
        np.testing.assert_allclose(value, permuted[field], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize(
    "replace,error",
    [
        ({"weights": [1, 1, 1, 0, 1, 1]}, "strictly positive"),
        ({"status": [0, 0, 0, 0, 0, 0]}, "at least one event"),
        ({"covariates": [1, 1, 1, 1, 1, 1]}, "nonconstant"),
        ({"psu": [1, 1, 1, 1, 1, 1]}, "only one non-certainty PSU"),
        ({"fpc": [0.1, 0.2, 0.1, 0.1, 0.1, 0.1]}, "constant within each stratum"),
        ({"ties": "exact"}, "efron"),
    ],
)
def test_invalid_inputs(replace, error):
    arguments = {
        "time": [1, 2, 3, 4, 5, 6],
        "status": [1, 0, 1, 0, 1, 1],
        "covariates": [1, 0, 0, 1, 0, 1],
        "weights": [1, 1, 1, 1, 1, 1],
    }
    with pytest.raises(ValueError, match=error):
        svycoxph(**(arguments | replace))
