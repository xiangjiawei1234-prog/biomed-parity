"""Compare exact binomial sequential calculations with R gsDesign."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.group_sequential_binomial import (
    binomial_conditional_probabilities,
    binomial_crossing_probabilities,
    binomial_efficacy_boundaries,
    binomial_harm_probabilities,
    binomial_sprt,
)

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    kind = case["kind"]
    options = case.get("options", {})
    if kind == "calibration":
        fit = binomial_efficacy_boundaries(case["sample_sizes"], case["null_probability"], **options)
        exact = binomial_crossing_probabilities(
            fit.sample_sizes, fit.lower_boundaries, fit.upper_boundaries, case["null_probability"],
        )
        first = exact.lower if options.get("direction") == "lower" else exact.upper
        np.testing.assert_allclose(fit.cumulative_error, first[:, 0].cumsum(), atol=1e-14, rtol=0)
    elif kind == "sprt":
        fit = binomial_sprt(**options)
    elif kind == "harm":
        fit = binomial_harm_probabilities(
            case["sample_sizes"], case["efficacy"], case["futility"], case["harm"], case["probabilities"],
            test_futility=case["test_futility"], test_harm=case["test_harm"],
        )
    else:
        inputs = (case["sample_sizes"], case["lower"], case["upper"], case["probabilities"])
        respect_lower = case.get("respect_lower", True)
        if kind == "conditional":
            fit = binomial_conditional_probabilities(
                *inputs, look=case["look"], successes=case["successes"], respect_lower=respect_lower,
            )
        else:
            fit = binomial_crossing_probabilities(*inputs, respect_lower=respect_lower)
    for field, expected in reference.items():
        if field == "id":
            continue
        actual = getattr(fit, field) if hasattr(fit, field) else getattr(fit.operating_characteristics, field)
        context = f"{case['id']} {field}"
        if field in {"sample_sizes", "lower_boundaries", "upper_boundaries"}:
            np.testing.assert_array_equal(actual, expected, err_msg=context)
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, err_msg=context)


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
    return {"method": "gsDesign exact binomial", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))
