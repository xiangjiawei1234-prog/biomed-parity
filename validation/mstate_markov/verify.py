"""Compare Markov test traces with R mstate using identical bootstrap draws."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.mstate_markov import markov_test, optimal_weights_matrix, optimal_weights_multiple
from biomed_parity.mstate_prep import msprep

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    records = msprep(
        case["time"], case["status"], case["transitions"],
        start_state=case.get("start_state", 0), start_time=case.get("start_time", 0),
        id=case.get("subject_ids"),
    )
    args = (records, case["transition"], case["grid"])
    options = dict(distribution=case["distribution"], multipliers=reference["multipliers"])
    result = markov_test(*args, **options)
    fields = {field: value for field, value in reference.items() if field not in ("id", "multipliers", "weights")}
    _assert_fields(case["id"], result, fields)
    deterministic = markov_test(*args, n_bootstrap=0)
    np.testing.assert_array_equal(deterministic.z_trace, result.z_trace)
    np.testing.assert_array_equal(deterministic.chi_squared_trace, result.chi_squared_trace)

    min_time = case.get("min_time", 0)
    state_weights = optimal_weights_matrix(*args)
    overall_weights = optimal_weights_multiple(*args, min_time=min_time)
    np.testing.assert_allclose(state_weights, reference["weights"]["state"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(overall_weights, reference["weights"]["overall"], rtol=0, atol=1e-12)
    optimal = markov_test(*args, **options, state_weights="optimal", overall_weights="optimal", min_time=min_time)
    _assert_fields(f"{case['id']} state-specific weights", optimal, reference["weights"]["optimal"])
    np.testing.assert_array_equal(optimal.z_trace, result.z_trace)
    np.testing.assert_array_equal(optimal.chi_squared_trace, result.chi_squared_trace)

    # The closures returned in mstate 0.3.3 all capture the final column.
    shared = state_weights[:, -1] * np.diff(case["grid"], prepend=min_time)
    r_functions = markov_test(*args, **options, state_weights=shared, overall_weights=overall_weights)
    _assert_fields(f"{case['id']} R's shared weights", r_functions, reference["weights"]["r_functions"])


def _assert_fields(case_id, result, fields):
    for field, expected in fields.items():
        actual = getattr(result, field)
        if field in ("qualifying_states", "state_statistics", "state_pvalues"):
            expected = np.atleast_1d(expected)
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-12, err_msg=f"{case_id} {field}",
        )


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
