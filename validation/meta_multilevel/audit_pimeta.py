"""Compare the optional pimeta package against the archived local R results."""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pimeta
import scipy


FIXTURES = Path(__file__).resolve().parent


def _absolute_difference(actual, expected):
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise ValueError("candidate returned a nonfinite vector or unexpected shape")
    return float(np.max(np.abs(actual - expected)))


def main():
    inputs = json.loads((FIXTURES / "cases.json").read_text())["datasets"]
    archive = json.loads((FIXTURES / "expected.json").read_text())
    cases = (
        ("nested", "nested-reml-profile-and-crve", "~ 1 | study/endpoint", "t", "contain"),
        ("crossed", "crossed-reml", ["~ 1 | study", "~ 1 | center"], "z", "residual"),
    )
    signature = inspect.signature(pimeta.rma_mv).parameters
    report = {
        "pimeta_version": pimeta.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "metafor_version": archive["package_version"],
        "rma_mv_fixed_variance_argument": "sigma2" in signature,
        "rma_mv_known_correlation_argument": "R" in signature,
        "cases": {},
    }
    for dataset, case_id, formula, test, dfs in cases:
        data = inputs[dataset]
        expected = archive["results"][case_id]
        v = np.asarray(data["variances"])
        groups = np.asarray(data["sampling_clusters"])
        V = data["sampling_correlation"] * np.sqrt(v[:, None] * v) * (
            groups[:, None] == groups
        )
        np.fill_diagonal(V, v)
        labels = {}
        for name, values in data["groups"].items():
            values = np.asarray(values)
            labels[name] = values[:, -1] if values.ndim == 2 else values
        # NumPy on macOS can signal spurious flags in BLAS matrix products.
        # The finite checks below still fail for nonfinite candidate output.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            fit = pimeta.rma_mv(
                data["effects"], V=V, mods=data["moderators"], data=pd.DataFrame(labels),
                random=formula, method="REML", test=test, dfs=dfs,
            )
            case = {
                "coefficient_max_abs_difference": _absolute_difference(fit.beta, expected["coefficients"]),
                "variance_max_abs_difference": _absolute_difference(
                    fit.sigma2, list(expected["variance_components"].values())
                ),
                "coefficient_covariance_max_abs_difference": _absolute_difference(
                    fit.vb, expected["covariance"]
                ),
                "log_likelihood_abs_difference": abs(fit.ll - expected["log_likelihood"]),
                "confint_keys": list(pimeta.confint(fit)),
            }
            if "CR2" in expected.get("robust", {}):
                robust = pimeta.robust(fit, cluster=groups, clubSandwich=True)
                target = expected["robust"]["CR2"]
                case["cr2_standard_error_max_abs_difference"] = _absolute_difference(
                    robust["se"], target["standard_errors"]
                )
                case["cr2_df_max_abs_difference"] = _absolute_difference(
                    robust["dfs"], target["degrees_of_freedom"]
                )
                case["cr2_result_fields"] = sorted(robust)
                try:
                    pimeta.profile_rma(fit)
                except (TypeError, ValueError, AttributeError, NotImplementedError) as error:
                    case["profile_rma_failure"] = f"{type(error).__name__}: {error}"
                else:
                    case["profile_rma_failure"] = None
        report["cases"][case_id] = case
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
