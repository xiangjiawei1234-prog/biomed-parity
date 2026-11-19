"""Compare weighted random-intercept REML against R variancePartition and lmerTest."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.dream import dream_random_intercept

HERE = Path(__file__).resolve().parent


def assert_case(case, reference):
    result = dream_random_intercept(**{key: value for key, value in case.items() if key != "id"})
    for field, value in reference.items():
        if field == "id":
            continue
        actual = getattr(result, field)
        rtol, atol = (5e-6, 0.0) if field == "pvalue" else (1e-6, 1e-8)
        np.testing.assert_allclose(
            actual, value, rtol=rtol, atol=atol, err_msg=f"{case['id']} {field}"
        )


def verify(rscript="Rscript"):
    reference = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    reference = json.loads(reference.stdout)
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    expected = {case["id"]: case for case in reference["cases"]}
    assert set(expected) == {case["id"] for case in cases}
    for case in cases:
        assert_case(case, expected[case["id"]])
    return {"method": "variancePartition::dream", "cases_passed": len(cases), **reference["versions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))


if __name__ == "__main__":
    main()
