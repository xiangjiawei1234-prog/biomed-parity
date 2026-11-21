"""Check binary logistic coefficients, penalty, and inference against R rms."""

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
    options = {key: case[key] for key in ("penalty", "penalty_matrix", "categorical_blocks") if key in case}
    result = fit_penalized_logistic(reference["x"], reference["y"], **options)
    for field in fields(result):
        name = field.name
        np.testing.assert_allclose(
            getattr(result, name), reference[name], atol=3e-9, rtol=2e-9,
            err_msg=f"{case['id']} {name}",
        )


def verify(rscript="Rscript"):
    process = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(process.stdout)
    reference = {row["id"]: row for row in report["cases"]}
    assert set(reference) == {case["id"] for case in CASES}
    for case in CASES:
        assert_case(case, reference[case["id"]])
    return {"method": "rms::lrm", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    print(json.dumps(verify(parser.parse_args().rscript)))
