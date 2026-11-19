from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.splines import (
    RestrictedCubicSpline, SplineKnotsWarning, fit_restricted_cubic_spline,
    restricted_cubic_spline_knots,
)

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "splines"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["cases"]


def _compare(actual, expected):
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=2e-11, atol=5e-12)


def _assert_case(case, reference):
    assert case["id"] == reference["id"]
    x = np.asarray(case["x"], dtype=float)
    kwargs = {key: value for key, value in case.items() if key not in ("id", "warning", "x", "xnew")}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", SplineKnotsWarning)
        fit = fit_restricted_cubic_spline(x, **kwargs)
    assert len(captured) == len(reference["warnings"]) == int(case.get("warning", False))
    assert all(isinstance(w.message, SplineKnotsWarning) for w in captured)
    _compare(fit.knots, reference["knots"])
    signs = 1
    if reference["pc_rotation"] is not None:
        _compare(fit.pc_center, reference["pc_center"])
        _compare(fit.pc_scale, reference["pc_scale"])
        rotation = np.asarray(fit.pc_rotation)
        signs = np.sign(np.sum(rotation * np.asarray(reference["pc_rotation"]), axis=0))
        _compare(rotation * signs, reference["pc_rotation"])
    _compare(fit.transform(x) * signs, reference["design"])
    _compare(fit.transform(case["xnew"]) * signs, reference["prediction"])
    if reference["rms_design"] is not None:
        _compare(fit.transform(x), reference["rms_design"])
    if reference["integral_derivative"] is not None:
        _compare(fit.transform(case["xnew"], derivative=1), reference["integral_derivative"])


@pytest.mark.parametrize(("case", "expected"), list(zip(CASES, EXPECTED, strict=True)), ids=[c["id"] for c in CASES])
def test_spline_design_matches_r(case, expected):
    _assert_case(case, expected)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    command = ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")]
    call = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    references = json.loads(call.stdout)["cases"]
    for case, reference in zip(CASES, references, strict=True):
        _assert_case(case, reference)


def test_derivatives_and_large_linear_tails():
    fit = fit_restricted_cubic_spline([0, 1, 2], knots=[0, 1, 2], norm=0)
    x = np.array([-1, 0.3, 1.3, 2.3, 4.9])
    step = 1e-5
    first = (fit.transform(x + step) - fit.transform(x - step)) / (2 * step)
    second = (fit.transform(x + step, derivative=1) - fit.transform(x - step, derivative=1)) / (2 * step)
    np.testing.assert_allclose(fit.transform(x, derivative=1), first, rtol=1e-9, atol=5e-10)
    np.testing.assert_allclose(fit.transform(x, derivative=2), second, rtol=1e-9, atol=5e-10)
    assert fit.nonlinear_columns == (False, True)
    assert fit.transform(1e16)[0, 1] == pytest.approx(6e16 - 6)
    np.testing.assert_array_equal(fit.transform([1e16, 1e17], derivative=2), np.zeros((2, 2)))


def test_saved_transform_and_missing_rows():
    training = np.linspace(1, 50, 120)
    fit = fit_restricted_cubic_spline(training, principal_components=True)
    restored = RestrictedCubicSpline(**asdict(fit))
    np.testing.assert_array_equal(restored.transform([4, 12]), fit.transform([4, 12]))
    assert np.isnan(restored.transform([np.nan])).all()
    assert np.isnan(restored.transform([np.nan], derivative=2)).all()
    assert all(restored.nonlinear_columns)


def test_knot_selection_precedes_missing_replacement():
    x = np.linspace(1, 50, 120)
    contaminated = np.r_[x, np.repeat(np.nan, 60)]
    fit = fit_restricted_cubic_spline(contaminated, missing_value=-3)
    _compare(fit.knots, restricted_cubic_spline_knots(x))
    np.testing.assert_array_equal(fit.transform([np.nan]), fit.transform([-3]))


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: restricted_cubic_spline_knots([0, 1, 2, 3, np.nan, 4]), "six nonmissing"),
        (lambda: restricted_cubic_spline_knots(np.arange(50), n_knots=2), "at least three"),
        (lambda: restricted_cubic_spline_knots(np.arange(50), tie_fraction=-1), "tie_fraction"),
        (lambda: fit_restricted_cubic_spline([0, 1, 2], knots=[0, 1, 1]), "distinct knots"),
        (lambda: fit_restricted_cubic_spline([0, 1, 2], knots=[0, 1, np.inf]), "infinite"),
        (lambda: fit_restricted_cubic_spline([0, 1, 2], knots=[0, 1, 2], norm=4), "norm"),
        (lambda: fit_restricted_cubic_spline([0, 1, 2, np.nan], knots=[0, 1, 2],
                                              principal_components=True), "finite training rows"),
    ],
)
def test_invalid_arguments(operation, message):
    with pytest.raises(ValueError, match=message):
        operation()
