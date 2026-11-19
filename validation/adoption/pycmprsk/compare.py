"""Compare native pycmprsk against a live R cmprsk session on generated data."""
import csv
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pycmprsk import crr, cuminc, predict_crr, timepoints

from biomed_parity.competing_risks import cmprsk_residuals_by_time, cmprsk_timepoints


def generate_data(path):
    rng = np.random.default_rng(820126)
    n = 144
    arm = np.tile(["A", "B", "C"], n // 3)
    stratum = np.repeat(["site_1", "site_2", "site_3"], n // 3)
    times = rng.integers(1, 30, size=n)
    status = rng.choice([2, 4, 9], size=n, p=[0.44, 0.28, 0.28])
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, size=n).astype(float)
    x1[40] = np.nan

    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time", "status", "arm", "stratum", "x1", "x2"])
        for values in zip(times, status, arm, stratum, x1, x2):
            writer.writerow([*values[:4], "NA" if np.isnan(values[4]) else values[4], values[5]])
    return times, status, arm, stratum, np.column_stack([x1, x2])


def time_effect(times):
    return np.log1p(times).reshape(-1, 1)


def calculate(times, status, arm, stratum, covariates):
    results = {}
    adapted = {}
    for rho in (0, 1):
        key = f"cuminc_{rho}"
        fitted = cuminc(times, status, group=arm, strata=stratum, rho=rho, cencode=9)
        results[f"{key}.tests"] = fitted.tests
        for label, curve in fitted.curves.items():
            for field in ("time", "est", "var"):
                results[f"{key}.{label}.{field}"] = getattr(curve, field)
        queries = [0, 7, 20, 35]
        requested = timepoints(fitted, queries)
        for i, label in enumerate(requested["labels"]):
            curve = fitted.curves[label]
            _, estimate, variance = cmprsk_timepoints(curve.time, curve.est, curve.var, queries)
            results[f"{key}.{label}.point_est"] = requested["est"][i]
            results[f"{key}.{label}.point_var"] = requested["var"][i]
            adapted[f"{key}.{label}.point_est"] = estimate
            adapted[f"{key}.{label}.point_var"] = variance

    predict_x = np.array([[-0.5, 0], [0.5, 1.0]])
    for varying in (False, True):
        key = "crr_time_varying" if varying else "crr_fixed"
        extra = {"cov2": covariates[:, :1], "tf": time_effect, "cengroup": arm} if varying else {}
        fitted = crr(times, status, cov1=covariates, failcode=2, cencode=9, **extra)
        predicted = predict_crr(fitted, cov1=predict_x, cov2=predict_x[:, :1] if varying else None)
        fields = {
            "coef": "coef", "var": "var", "res": "res", "loglik": "loglik",
            "loglik.null": "loglik_null", "uftime": "uftime", "converged": "converged",
            "n": "n", "n.missing": "n_missing",
        }
        for r_field, py_field in fields.items():
            results[f"{key}.{r_field}"] = getattr(fitted, py_field)
        if not fitted.converged:
            raise RuntimeError(f"{key} did not converge")
        results[f"{key}.prediction"] = predicted
        primary_events = (status == 2) & np.isfinite(covariates).all(axis=1)
        event_times = np.sort(times[primary_events])
        residual_times, residuals = cmprsk_residuals_by_time(event_times, fitted.res)
        np.testing.assert_array_equal(residual_times, fitted.uftime)
        adapted[f"{key}.res"] = residuals
    return results, {**results, **adapted}


def main():
    reference = Path(__file__).with_name("reference.R")
    with TemporaryDirectory(prefix="validate-pycmprsk-") as temp:
        fixture = Path(temp) / "synthetic_cohort.csv"
        data = generate_data(fixture)
        process = subprocess.run(
            ["Rscript", "--vanilla", str(reference), str(fixture)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    expected = {}
    for line in process.stdout.splitlines():
        key, values = line.split("\t", maxsplit=1)
        expected[key] = np.array([float("nan" if value == "NA" else value) for value in values.split(",")])

    raw, adapted = calculate(*data)
    if expected.keys() != raw.keys():
        raise AssertionError(f"Field mismatch: {sorted(expected.keys() ^ raw.keys())}")

    raw_differences = []
    max_difference = 0.0
    for key, expected_value in expected.items():
        direct = np.asarray(raw[key], dtype=float).ravel()
        corrected = np.asarray(adapted[key], dtype=float).ravel()
        if direct.shape != expected_value.shape or not np.allclose(
            direct, expected_value, rtol=1e-7, atol=1e-9, equal_nan=True
        ):
            raw_differences.append(key)
        np.testing.assert_array_equal(corrected.shape, expected_value.shape, err_msg=key)
        np.testing.assert_allclose(
            corrected, expected_value, rtol=1e-7, atol=1e-9, equal_nan=True, err_msg=key
        )
        finite = np.isfinite(expected_value)
        if finite.any():
            field_error = np.max(np.abs(corrected[finite] - expected_value[finite]))
            max_difference = max(max_difference, float(field_error))

    print(f"Python {platform.python_version()}, pycmprsk {version('pycmprsk')}")
    print(f"Unmodified pycmprsk: {len(raw) - len(raw_differences)}/{len(raw)} matching fields.")
    print("Unmodified pycmprsk differences:", ", ".join(raw_differences))
    print(f"With adapters: {len(adapted)}/{len(raw)} matching fields; "
          f"maximum absolute difference {max_difference:.3g}.")


if __name__ == "__main__":
    main()
