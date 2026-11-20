"""Compare predictor summaries and Wald predictions with R rms and multcomp."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit, logit

from biomed_parity.rms_prediction import datadist, predict
from biomed_parity.splines import fit_restricted_cubic_spline

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def _compare_summary(case, reference):
    if case.get("preset") == "varying_n":
        data = {"x": np.arange(1, 301), "z": np.r_[np.arange(1, 226)**2 / 100, np.full(75, np.nan)]}
    else:
        data = case["data"]
    summary = datadist(data, factor_levels=case.get("factor_levels"), **case.get("options", {}))
    for name, expected in reference["limits"].items():
        limits = summary.limits[name]
        low, high = (None, None) if limits.effect is None else limits.effect
        actual = [low, limits.adjust_to, high, *limits.display, *limits.overall]
        if limits.levels is None:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11, err_msg=f"{case['id']} {name}")
        else:
            assert actual == expected
    expected_values = reference["values"] or {}
    assert set(summary.values) == set(expected_values)
    for name in summary.values:
        np.testing.assert_array_equal(summary.values[name], np.atleast_1d(expected_values[name]))


def _compare_prediction(case, reference, report):
    data = report["predictors"]
    summary = datadist(data, factor_levels=report["factor_levels"])
    spline = fit_restricted_cubic_spline(data["x"], knots=report["knots"])

    def design(grid):
        x = spline.transform(grid["x"])
        standard = grid["arm"] == "standard"
        targeted = grid["arm"] == "targeted"
        features = [x, standard, targeted, grid["z"]]
        if case["model"] == "ols":
            features.extend([x * standard[:, None], x * targeted[:, None]])
        if case["model"] != "cph":
            features.insert(0, np.ones(len(x)))
        return np.column_stack(features)

    options = dict(case.get("options", {}))
    scale = options.pop("transform", None)
    transform = {None: None, "logistic": expit, "exp": np.exp}[scale]
    result = predict(
        summary, case["vary"], design, reference["coefficients"], reference["covariance"],
        center=reference["center"], residual_df=reference["residual_df"], transform=transform, **options,
    )
    assert tuple(result.grid) == tuple(reference["grid"])
    for name, actual in result.grid.items():
        expected = reference["grid"][name]
        if actual.dtype.kind in "iuf":
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=f"{case['id']} {name}")
        else:
            np.testing.assert_array_equal(actual, expected)
    for field in ("linear_predictor", "standard_error", "estimate"):
        np.testing.assert_allclose(
            getattr(result, field), reference[field], rtol=0, atol=1e-10, err_msg=f"{case['id']} {field}",
        )
    if options.get("confidence_type") == "simultaneous":
        inverse = {None: np.asarray, "logistic": logit, "exp": np.log}[scale]
        error = result.standard_error
        nonzero = error > 0
        for field in ("lower", "upper"):
            actual, expected = getattr(result, field), np.array(reference[field])
            np.testing.assert_allclose(
                actual[~nonzero], expected[~nonzero], rtol=0, atol=1e-10,
                err_msg=f"{case['id']} zero-variance {field}",
            )
            np.testing.assert_allclose(
                (inverse(actual[nonzero]) - inverse(expected[nonzero])) / error[nonzero],
                0, rtol=0, atol=1.5e-3, err_msg=f"{case['id']} standardized {field}",
            )
        if "rms_critical" in reference:
            j = error.argmax()
            critical = (inverse(result.upper[j]) - inverse(result.lower[j])) / (2 * error[j])
            np.testing.assert_allclose(
                critical, reference["rms_critical"], rtol=0, atol=6e-3,
                err_msg=f"{case['id']} rms simultaneous critical value",
            )
    else:
        # SciPy 1.12's inverse t CDF differs from R at the last few digits.
        tolerance = 2e-10 if reference["residual_df"] is not None else 1e-10
        for field in ("lower", "upper"):
            np.testing.assert_allclose(
                getattr(result, field), reference[field], rtol=0, atol=tolerance, err_msg=f"{case['id']} {field}",
            )


def assert_case(case, reference, report):
    if case["kind"] == "summary":
        _compare_summary(case, reference)
    else:
        _compare_prediction(case, reference, report)


def verify(rscript="Rscript"):
    proc = subprocess.run(
        [rscript, "--vanilla", str(HERE / "oracle.R"), str(HERE / "cases.json")],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(proc.stdout)
    reference = {item["id"]: item for item in report["cases"]}
    assert set(reference) == {item["id"] for item in CASES}
    for case in CASES:
        assert_case(case, reference[case["id"]], report)
    return {
        "method": "rms::datadist, rms::Predict, and multcomp::glht",
        "cases_passed": len(CASES),
        **report["versions"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    print(json.dumps(verify(args.rscript)))
