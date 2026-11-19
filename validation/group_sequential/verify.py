"""Compare Python efficacy boundaries against the installed R gsDesign package."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.group_sequential import (
    conditional_power,
    crossing_probabilities,
    efficacy_boundaries,
    efficacy_futility_boundaries,
    symmetric_boundaries,
)

HERE = Path(__file__).resolve().parent


def assert_case(case, expected):
    if "design" in case:
        _assert_two_boundary_case(case, expected)
        return
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


def _assert_two_boundary_case(case, expected):
    inputs = {key: value for key, value in case.items() if key not in {"id", "design", "interim_z"}}
    design = symmetric_boundaries if case["design"] == "symmetric" else efficacy_futility_boundaries
    result = design(**inputs)
    for field in ("upper_boundaries", "lower_boundaries", "alternative_effect"):
        if expected[field] is None:
            assert getattr(result, field) is None
        else:
            np.testing.assert_allclose(
                getattr(result, field), expected[field], rtol=0, atol=2e-6, err_msg=f"{case['id']} {field}"
            )
    for field in ("information_fractions", "cumulative_alpha", "cumulative_beta"):
        if expected[field] is None:
            assert getattr(result, field) is None
        else:
            np.testing.assert_allclose(
                getattr(result, field), expected[field], rtol=0, atol=1e-14, err_msg=f"{case['id']} {field}"
            )

    # Use the same thresholds as R to isolate the probability algorithms.
    args = (
        expected["information_fractions"],
        expected["lower_boundaries"],
        expected["upper_boundaries"],
    )
    for comparison in expected["evaluations"]:
        kwargs = {"effect": comparison["effect"], "respect_lower": comparison["respect_lower"]}
        if comparison["look"] is None:
            observed = crossing_probabilities(*args, **kwargs)
        else:
            observed = conditional_power(*args, look=comparison["look"], z=comparison["z"], **kwargs)
        for field in ("lower", "upper"):
            np.testing.assert_allclose(
                getattr(observed, field), comparison[field], rtol=0, atol=5e-10,
                err_msg=f"{case['id']} {field} {comparison}",
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
