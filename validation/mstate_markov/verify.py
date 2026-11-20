"""Compare Markov test traces with R mstate using identical bootstrap draws."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.mstate_markov import markov_test
from biomed_parity.mstate_prep import msprep

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    records = msprep(
        case["time"], case["status"], case["transitions"],
        start_state=case.get("start_state", 0), start_time=case.get("start_time", 0),
        id=case.get("subject_ids"),
    )
    result = markov_test(
        records, case["transition"], case["grid"],
        distribution=case["distribution"], multipliers=reference["multipliers"],
    )
    for field, expected in reference.items():
        if field in ("id", "multipliers"):
            continue
        actual = getattr(result, field)
        if field in ("qualifying_states", "state_statistics", "state_pvalues"):
            expected = np.atleast_1d(expected)
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-12, err_msg=f"{case['id']} {field}",
        )
    deterministic = markov_test(records, case["transition"], case["grid"], n_bootstrap=0)
    np.testing.assert_array_equal(deterministic.z_trace, result.z_trace)
    np.testing.assert_array_equal(deterministic.chi_squared_trace, result.chi_squared_trace)


def verify(rscript="Rscript"):
    proc = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(proc.stdout)
    reference = {item["id"]: item for item in report["cases"]}
    assert set(reference) == {item["id"] for item in CASES}
    for case in CASES:
        assert_case(case, reference[case["id"]])
    return {"method": "mstate::MarkovTest", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))
