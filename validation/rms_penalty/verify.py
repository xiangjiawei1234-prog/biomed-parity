"""Verify binary logistic penalty grids and selection against rms::pentrace."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from biomed_parity.rms_penalty import trace_logistic_penalties

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def assert_case(case, reference):
    options = {name: value for name, value in case.items() if name not in ("id", "design", "penalties")}
    result = trace_logistic_penalties(reference["x"], reference["y"], case["penalties"], keep_coefficients=True, **options)
    assert result.penalty_names == tuple(reference["penalty_names"])
    assert result.criterion == reference["criterion"]
    assert result.best_index == reference["best_index"]
    np.testing.assert_array_equal(result.penalties, reference["penalties"])
    selected = reference["penalties"][reference["best_index"]]
    assert result.best_penalty == (
        dict(zip(result.penalty_names, selected, strict=True)) if isinstance(case["penalties"], dict) else selected[0]
    )
    for name in ("effective_df", "likelihood_ratio", "aic", "bic", "aicc", "objective", "adjusted_covariance", "coefficients"):
        np.testing.assert_allclose(
            getattr(result, name), reference[name], atol=3e-9, rtol=2e-9, err_msg=f"{case['id']} {name}",
        )
    for name in ("coefficients", "covariance", "effective_df_diagonal"):
        np.testing.assert_allclose(
            getattr(result.best_fit, name), reference[f"best_{name}"],
            atol=3e-9, rtol=2e-9, err_msg=f"{case['id']} selected {name}",
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
    return {"method": "rms::pentrace", "cases_passed": len(CASES), **report["versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    print(json.dumps(verify(parser.parse_args().rscript)))
