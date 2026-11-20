"""Compare repeated and sequential inference with R gsDesign."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.group_sequential import efficacy_boundaries, symmetric_boundaries
from biomed_parity.group_sequential_inference import (
    repeated_confidence_intervals,
    repeated_pvalues,
    sequential_pvalue,
)

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    options = dict(case.get("options", {}))
    if case.get("kind") == "interval":
        alternative = options.pop("alternative", "greater")
        if alternative == "two-sided":
            critical = symmetric_boundaries(case["fractions"], **options).upper_boundaries
        else:
            critical = efficacy_boundaries(case["fractions"], **options).z_boundaries
        result = repeated_confidence_intervals(
            case["estimates"], case["standard_errors"], critical, alternative=alternative
        )
        np.testing.assert_allclose(result.lower, reference["lower"], atol=1e-7, rtol=0)
        if reference["upper"] is None:
            assert np.isposinf(result.upper).all()
        else:
            np.testing.assert_allclose(result.upper, reference["upper"], atol=1e-7, rtol=0)
        return
    result = repeated_pvalues(case["information"], case["z"], **options)
    direct = sequential_pvalue(case["information"], case["z"], **options)
    # The reference root solver uses R's default tolerance on the Z scale.
    np.testing.assert_allclose(result.repeated, np.atleast_1d(reference["repeated"]), atol=1e-5, rtol=0)
    np.testing.assert_allclose(direct, reference["sequential"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(direct, result.sequential, atol=1e-11, rtol=0)


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
    return {"method": "gsDesign::sequentialPValue and repeated intervals", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))
