from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.meta_multilevel import meta_multilevel
from biomed_parity.meta_prediction import predict_multilevel_meta


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "meta_prediction"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _fit(case, covariances=None):
    groups = np.repeat(np.arange(7), 3)
    j = np.arange(21)
    x = groups / 3 - 1
    z = np.tile([-1, 0, 1], 7)
    vi = 0.024 + ((j * 13) % 7) / 150
    y = 0.6 + 0.25 * x + 0.42 * z + 0.4 * np.cos(groups * 2) + 0.15 * np.sin(j * 3)
    covariance = 0.23 * np.outer(np.sqrt(vi), np.sqrt(vi)) * (groups[:, None] == groups)
    np.fill_diagonal(covariance, vi)
    fixed = {"group": case["fix_variance"]} if "fix_variance" in case else {}
    if covariances:
        fixed.update(extra=0.07)
    return meta_multilevel(
        y, covariance, moderators=np.column_stack((x, z)), random_intercepts={"group": groups},
        test=case["test"], dfs=case.get("dfs", "residual"), fixed_variances=fixed,
        covariance_components=covariances,
    )


def _check(case, reference):
    fit = _fit(case)
    options = {key: case[key] for key in (
        "prediction", "confidence_level", "adjust", "heterogeneity", "new_sampling_variance",
    ) if key in case}
    if case.get("transform") == "exp":
        options["transform"] = np.exp
    result = predict_multilevel_meta(fit, case.get("rows"), covariance=True, **options)
    for name, actual in asdict(result).items():
        expected = reference[name]
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), atol=6e-7, rtol=3e-6,
                                       err_msg=name)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_metafor_prediction_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_metafor_prediction():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    expected = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, expected[case["id"]])


def test_nonunit_random_effect_diagonal():
    extra = np.linspace(0.1, 1, 21)
    fit = _fit({"test": "z", "fix_variance": 0.1}, covariances={"extra": np.diag(extra)})
    predictions = predict_multilevel_meta(fit)
    np.testing.assert_allclose(predictions.prediction_standard_errors**2,
                               predictions.standard_errors**2 + 0.1 + 0.07 * extra)
    with pytest.raises(ValueError, match="nonconstant fitted random variance"):
        predict_multilevel_meta(fit, [1, 0, 1])
    result = predict_multilevel_meta(fit, [1, 0, 1], heterogeneity=0.5)
    np.testing.assert_allclose(result.prediction_standard_errors**2 - result.standard_errors**2, 0.5)
