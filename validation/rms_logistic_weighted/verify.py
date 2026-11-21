"""Verify weighted binary logistic fits and offset inference against rms."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.rms_logistic import fit_penalized_logistic

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    result = fit_penalized_logistic(
        reference["x"], reference["y"], penalty=case.get("penalty", 0),
        weights=reference["input_weights"], normwt=case.get("normwt", False), offset=reference["input_offset"],
    )
    for field in fields(result):
        name = field.name
        expected = reference[name]
        if name == "penalty_matrix" and case.get("intercept_only"):
            expected = np.asarray(expected).reshape(0, 0)
        np.testing.assert_allclose(
            getattr(result, name), expected, atol=2e-9, rtol=2e-9, err_msg=f"{case['id']} {name}",
        )


def verify(rscript="Rscript"):
    process = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(process.stdout)
    references = {row["id"]: row for row in report["cases"]}
    assert set(references) == {case["id"] for case in CASES}
    for case in CASES:
        assert_case(case, references[case["id"]])
    return {"method": "rms::lrm and lrm.fit with weights and offsets", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    print(json.dumps(verify(parser.parse_args().rscript)))
