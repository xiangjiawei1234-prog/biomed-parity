"""Compare interval regression bootstrap inference with R icenReg fixed-draw refits."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.interval_bootstrap import bootstrap_interval_regression

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    right = [np.inf if value is None else value for value in reference["right"]]
    result = bootstrap_interval_regression(
        reference["left"], right, reference["design"], model=case["model"],
        weights=reference["weights"] if case.get("weighted") else None,
        bounds=case.get("bounds", (False, True)), bootstrap_indices=reference["bootstrap_indices"],
        confidence_level=case.get("confidence_level", 0.95),
    )
    assert result.failed_replicates == tuple(reference["failed_replicates"])
    np.testing.assert_allclose(result.fit.coefficients, reference["coefficients"], atol=3e-5, rtol=0)
    for field in ("replicates", "covariance", "standard_error", "z", "pvalue", "lower", "upper"):
        np.testing.assert_allclose(
            getattr(result, field), reference[field], atol=3e-5, rtol=0, err_msg=f"{case['id']} {field}",
        )


def verify(rscript="Rscript"):
    process = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True, capture_output=True, text=True, timeout=90,
    )
    report = json.loads(process.stdout)
    references = {row["id"]: row for row in report.pop("cases")}
    assert set(references) == {case["id"] for case in CASES}
    for case in CASES:
        assert_case(case, references[case["id"]])
    return {"method": "icenReg::ic_sp case bootstrap", "cases_passed": len(CASES), **report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    print(json.dumps(verify(parser.parse_args().rscript)))
