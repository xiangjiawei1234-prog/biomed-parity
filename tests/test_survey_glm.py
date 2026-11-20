import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from biomed_parity.survey_glm import svyglm
from biomed_parity.survey_replicates import svrepdesign

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "survey_glm"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _fixture(case):
    i = np.arange(1, 49)
    cluster = (i - 1) // 6
    x1 = ((17 * i) % 37 - 18) / 15
    x2 = ((13 * i) % 29 - 14) / 9
    offset = ((3 * i) % 11 - 5) / 18 if case.get("offset") else np.zeros(i.size)
    if case["family"] == "gaussian":
        y = 125 + 4.3 * x1 - 2.1 * x2 + offset + 3.1 * np.sin(0.9 * i) + 1.2 * np.cos(0.41 * i)
    else:
        y = (((29 * i) % 47) / 47 < expit(-0.3 + 0.65 * x1 - 0.5 * x2 + offset)).astype(float)
    weights = 1.2 + ((13 * i) % 17) / 8
    if case.get("zero_weights"):
        weights[np.isin(i, (6, 18))] = 0

    if case["scheme"] == "jk1":
        repweights = (cluster[:, None] != np.arange(8)) * (8 / 7)
        design_options = {"type": "JK1", "scale": 0.75}
    elif case["scheme"] == "fay":
        stratum = cluster // 2 + 1
        bits = np.array([number.bit_count() % 2 for number in range(8)])
        signs = bits[stratum[:, None] & np.arange(8)]
        repweights = np.where(signs == cluster[:, None] % 2, 1.6, 0.4)
        design_options = {"type": "Fay", "rho": 0.4}
    else:
        draw = np.arange(1, 9)
        repweights = 0.3 + ((i[:, None] * draw + 2 * draw + 3 * i[:, None]) % 11) / 5
        design_options = {"type": "bootstrap", "bootstrap_average": 2}

    if case.get("bad_replicate"):
        repweights[:, 0] = i <= 2
    if case.get("missing"):
        x1[np.isin(i, (5, 9, 22, 32))] = np.nan
        y[i == 37] = np.nan
        offset[i == 17] = np.nan
    if case.get("missing_rank"):
        repweights[:, 7] = repweights[:, 6] + (i == 3)
        y[i == 3] = np.nan
    combined = case.get("combined", False)
    if combined:
        repweights *= weights[:, None]
    if case.get("zero_sampling_weight_only"):
        weights[i == 2] = 0
    design = svrepdesign(
        repweights, weights, combined_weights=combined, rscales=case.get("rscales"),
        mse=case.get("mse", False), degf=case.get("degf"), **design_options,
    )
    n_covariates = case.get("n_covariates", 2)
    return {
        "values": y,
        "covariates": np.column_stack((x1, x2))[:, :n_covariates] if n_covariates else None,
        "design": design,
        "family": case["family"],
        "intercept": case.get("intercept", True),
        "offset": offset if case.get("offset") else None,
        "subset": (i % 7 != 1) & (i <= 43) if case.get("subset") else None,
        "rescale": case.get("rescale"),
        "df_resid": float(case["df_resid"]) if "df_resid" in case else None,
        "confidence_level": case.get("confidence_level", 0.95),
    }


def _check_case(case, expected):
    warning = pytest.warns(RuntimeWarning, match="1 invalid replicate") if case.get("bad_replicate") else nullcontext()
    with warning:
        actual = asdict(svyglm(**_fixture(case)))
    assert actual.keys() == expected.keys()
    np.testing.assert_array_equal(actual["included"], expected["included"])
    for field, value in actual.items():
        reference = np.asarray(expected[field], dtype=object)
        if field == "replicate_estimates":
            reference = np.where(reference == "NA", np.nan, reference).astype(float)
            # R preserves estimable coefficients in a rank-deficient replicate.
            reference[~np.isfinite(reference).all(axis=1)] = np.nan
        assert np.shape(value) == reference.shape, field
        np.testing.assert_allclose(
            value, reference.astype(float), rtol=1e-7, atol=1e-9, err_msg=f"{case['name']}.{field}",
        )


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_matches_recorded_r(case):
    _check_case(case, EXPECTED[case["name"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    results = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_case(case, results[case["name"]])


def test_combined_weights_and_row_order_with_domain_and_missing_data():
    case = next(case for case in CASES if case.get("subset"))
    arguments = _fixture(case)
    design = arguments["design"]
    order = np.random.default_rng(8061).permutation(design.sampling_weights.size)
    reordered_design = svrepdesign(
        7 * design.analysis_weights[order], 7 * design.sampling_weights[order],
        type=design.type, scale=design.scale, rscales=design.rscales, mse=design.mse,
    )
    reordered = {field: arguments[field][order] for field in ("values", "covariates", "offset", "subset")}
    result = asdict(svyglm(**arguments))
    other = asdict(svyglm(**(arguments | reordered | {"design": reordered_design})))
    fitted = np.full(order.size, np.nan)
    fitted[order[other["included"]]] = other["fitted_values"]
    other["fitted_values"] = fitted[result["included"]]
    other["included"] = other["included"][np.argsort(order)]
    for field, value in result.items():
        np.testing.assert_allclose(value, other[field], rtol=1e-10, atol=1e-11, err_msg=field)


def test_discards_zero_weight_replicate_without_rescaling_valid_replicates():
    design = svrepdesign([[0, 1, 1], [0, 2, 1], [0, 1, 2]], type="other", scale=0.5, mse=True)
    with pytest.warns(RuntimeWarning, match="1 invalid replicate"):
        fit = svyglm([1, 3, 2], None, design)
    np.testing.assert_allclose(fit.replicate_estimates[:, 0], [np.nan, 2.25, 2])
    np.testing.assert_allclose(fit.covariance, [[0.5 * 0.25**2]])


@pytest.mark.parametrize(
    "options, message",
    [
        ({"values": [1, 0, 2, 0], "family": "quasibinomial"}, "must be binary"),
        ({"values": [1, 0, np.nan, 0], "na_rm": False}, "missing values"),
        ({"offset": [np.nan] * 4}, "no complete rows"),
        ({"covariates": [1, 1, 1, 1]}, "rank deficient"),
        ({"family": "poisson"}, "family must be"),
        ({"family": "quasibinomial", "maxit": 1}, "did not converge"),
        ({"design": svrepdesign([[0, 0]] * 4)}, "no replicate"),
        ({"design": svrepdesign([[1, 1]] * 4, [1, 1, -1, 1])}, "nonnegative"),
    ],
)
def test_invalid_fit(options, message):
    arguments = {"values": [0, 1, 0, 1], "covariates": [1, 2, 3, 4], "design": svrepdesign([[1, 1]] * 4)}
    with pytest.raises(ValueError, match=message):
        svyglm(**(arguments | options))
