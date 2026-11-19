"""Compare transition probabilities and covariance against R mstate."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.multistate import probtrans

HERE = Path(__file__).resolve().parent


def assert_case(case, reference):
    fit = probtrans(**{key: value for key, value in case.items() if key != "id"})
    for name in ("times", "probabilities", "standard_errors", "covariance"):
        actual = getattr(fit, name)
        expected = reference[name]
        if expected is None:
            assert actual is None, name
        elif name == "times":
            np.testing.assert_array_equal(actual, expected, err_msg=case["id"])
        else:
            np.testing.assert_allclose(
                actual, expected, rtol=1e-12, atol=1e-13, err_msg=f"{case['id']} {name}"
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
    return {"method": "mstate::probtrans", "cases_passed": len(inputs), **reference["versions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))


if __name__ == "__main__":
    main()
