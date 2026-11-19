import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.rmst import rmst2

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "rmst"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _compare(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _compare(value, reference, f"{path}[{index}]")
    else:
        tolerance = 5e-7 if "adjusted" in path else 1e-10
        np.testing.assert_allclose(
            actual, float(expected), rtol=tolerance, atol=1e-10, err_msg=path
        )


def _check_case(case, expected):
    arguments = {key: value for key, value in case.items() if key != "id"}
    _compare(asdict(rmst2(**arguments)), expected, case["id"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_matches_recorded_r(case):
    _check_case(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    results = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_case(case, results[case["id"]])


@pytest.mark.parametrize(
    "times,status,arms,options,error",
    [
        ([1, 2], [1], [0, 1], {}, "same length"),
        ([1, 2], [1, 0], [0, 0], {}, "each contain"),
        ([1, 2], [2, 0], [0, 1], {}, "status"),
        ([-1, 2], [1, 1], [0, 1], {}, "nonnegative"),
        ([np.inf, 2], [1, 1], [0, 1], {}, "finite"),
        ([1, 2], [0, 1], [0, 1], {"tau": 1.2}, "tau"),
        ([1, 2], [1, 1], [0, 1], {"alpha": 1}, "alpha"),
        ([1, 2], [1, 1], [0, 1], {"covariates": [0.4]}, "covariates"),
        ([0, 2], [1, 1], [0, 1], {"covariates": [0.4, 0.3]}, "positive follow-up"),
        ([1, 2], [1, 1], [0, 1], {"covariates": [0.4, 0.3]}, "rank deficient"),
    ],
)
def test_invalid_inputs(times, status, arms, options, error):
    with pytest.raises(ValueError, match=error):
        rmst2(times, status, arms, **options)
