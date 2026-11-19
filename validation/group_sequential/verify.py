"""Compare Python efficacy boundaries against the installed R gsDesign package."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.group_sequential import efficacy_boundaries

HERE = Path(__file__).resolve().parent


def assert_case(case, expected):
    inputs = {key: value for key, value in case.items() if key != "id"}
    result = efficacy_boundaries(**inputs)
    tolerances = {
        "information_fractions": 1e-15,
        "z_boundaries": 3e-8,
        "nominal_pvalues": 1e-9,
        "cumulative_alpha": 1e-14,
        "incremental_alpha": 1e-14,
        "null_crossing_probabilities": 1e-10,
    }
    if case["id"] == "ldof_small_first_spending":
        # R 3.11.0 obtains early spending with 1 - pnorm(), which loses digits.
        assert abs(result.z_boundaries[0] - expected["z_boundaries"][0]) < 2e-5
    if case["id"] == "close_looks":
        # gsDesign's grid converges more slowly for tightly spaced analyses.
        tolerances.update(z_boundaries=1e-5, nominal_pvalues=8e-8)
    for field, tolerance in tolerances.items():
        actual = getattr(result, field)
        reference = expected[field]
        if case["id"] == "ldof_small_first_spending" and field == "z_boundaries":
            actual, reference = actual[1:], reference[1:]
        np.testing.assert_allclose(
            actual, reference, rtol=0, atol=tolerance, err_msg=f"{case['id']} {field}"
        )


def verify(rscript="Rscript"):
    reference = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    reference = json.loads(reference.stdout)
    inputs = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    expected = {case["id"]: case for case in reference["cases"]}
    assert set(expected) == {case["id"] for case in inputs}
    for case in inputs:
        assert_case(case, expected[case["id"]])
    return {"method": "gsDesign::gsDesign", "cases_passed": len(inputs), **reference["versions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))


if __name__ == "__main__":
    main()
