import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.imputation import pool_scalar

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "imputation"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _check_case(case, expected):
    actual = asdict(pool_scalar(**{key: value for key, value in case.items() if key != "id"}))
    assert actual.keys() == expected.keys()
    for field, value in actual.items():
        reference = np.nan if expected[field] == "NA" else expected[field]
        np.testing.assert_allclose(
            value, np.asarray(reference, dtype=float),
            rtol=1e-10, atol=5e-11, err_msg=f"{case['id']}.{field}",
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


@pytest.mark.parametrize(
    "q,u,options,error",
    [
        ([1], [0.2], {}, "at least two"),
        ([1, 2], [0.2, -0.3], {}, "nonnegative"),
        ([1, np.nan], [0.2, 0.3], {}, "finite"),
        ([1, 2], [0.2, 0.3], {"k": 3, "n": 2}, "n must exceed"),
        ([1, 2], [0.2, 0.3], {"confidence_level": 0}, "confidence_level"),
        ([1, 2], [0.2, 0.3], {"rule": "average"}, "rule"),
    ],
)
def test_invalid_inputs(q, u, options, error):
    with pytest.raises(ValueError, match=error):
        pool_scalar(q, u, **options)
