from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.imputation import pool_d1, pool_d2, pool_parameters


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "pooled_imputation"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _check_case(case, reference):
    args = {key: value for key, value in case.items() if key != "id"}
    kind = "d2" if "statistics" in args else "d1"
    test = pool_d2(**args) if kind == "d2" else pool_d1(**args)
    for name, value in asdict(test).items():
        # testConstraints itself uses numerical differentiation to transform covariance.
        np.testing.assert_allclose(value, float(reference[kind][name]), rtol=5e-8, atol=3e-11)
        np.testing.assert_allclose(value, float(reference[f"{kind}_direct"][name]),
                                   rtol=2e-10, atol=3e-11)
    if kind == "d2":
        return
    summary = pool_parameters(
        args["estimates"], args["covariances"], complete_df=args.get("complete_df", np.inf),
    )
    expected = reference["parameters"]
    for name in ("estimate", "within_covariance", "between_covariance", "total_covariance"):
        np.testing.assert_allclose(getattr(summary, name), expected[name], rtol=2e-13, atol=2e-14)
    for actual, ref in zip(summary.marginal, expected["marginal"], strict=True):
        for name, target in ref.items():
            value = actual.qbar if name == "estimate" else getattr(actual, name)
            np.testing.assert_allclose(value, float(target), rtol=2e-10, atol=3e-11, err_msg=name)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_mitml(case):
    _check_case(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_live_mitml():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    reference = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_case(case, reference[case["id"]])


def test_joint_wald_is_invariant_to_invertible_contrast_basis():
    args = {k: v for k, v in CASES[0].items() if k != "id"}
    transform = np.array([[2, 1], [-3, .5]])
    ordinary = pool_d1(**args)
    alternate = pool_d1(**(args | {
        "contrast": transform @ np.asarray(args["contrast"]),
        "null": transform @ np.asarray(args["null"]),
    }))
    assert asdict(ordinary) == pytest.approx(asdict(alternate))


def test_wald_rejects_singular_contrast_covariance():
    with pytest.raises(ValueError, match="contrast covariance is singular"):
        pool_d1([[1, 2], [2, 1]], [[[1, 1], [1, 1]]] * 2)


@pytest.mark.parametrize("function,kwargs,message", [
    (pool_parameters, {"covariances": [[[1, 0], [1, 1]]] * 2}, "symmetric"),
    (pool_parameters, {"covariances": [[[1, 2], [2, 1]]] * 2}, "semidefinite"),
    (pool_d1, {"contrast": [[1, 0], [2, 0]]}, "independent"),
    (pool_d1, {"null": [0, 1, 2]}, "contrast rows"),
    (pool_d1, {"complete_df": 0}, "positive"),
])
def test_invalid_multivariate_inputs(function, kwargs, message):
    data = {"estimates": [[1, 2], [2, 1]], "covariances": [np.eye(2)] * 2}
    with pytest.raises(ValueError, match=message):
        function(**(data | kwargs))
