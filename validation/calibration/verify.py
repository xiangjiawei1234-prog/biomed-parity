"""Compare binary logistic bootstrap metrics against the installed R rms package."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.calibration import validate_logistic

HERE = Path(__file__).resolve().parent


def assert_case(case, reference):
    result = validate_logistic(
        case["x"],
        case["y"],
        bootstrap_indices=reference["bootstrap_indices"],
        confidence=case["confidence"],
        emax_limits=case.get("emax_limits", (0, 1)),
    )
    assert result.metric_names == tuple(reference["metric_names"])
    np.testing.assert_array_equal(result.counts, reference["counts"], err_msg=case["id"])
    for column in ("original", "training", "test", "optimism", "corrected", "lower", "upper"):
        value = getattr(result, column)
        expected = reference[column]
        if expected is None:
            assert value is None, column
        else:
            np.testing.assert_allclose(
                value, expected, rtol=1e-8, atol=5e-9, err_msg=f"{case['id']} {column}"
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
    return {"method": "rms::validate.lrm", "cases_passed": len(inputs), **reference["versions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))


if __name__ == "__main__":
    main()
