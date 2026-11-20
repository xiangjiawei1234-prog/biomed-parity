import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.count_predictions import predict_count_mixed, simulate_count_mixed
from biomed_parity.zero_inflated import fit_zinb2_random_intercept, fit_zip_random_intercept


VALIDATION = Path(__file__).resolve().parents[1] / "validation"
FIXTURES = VALIDATION / "count_predictions"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _model(case):
    poisson = case["family"] == "poisson"
    if poisson:
        with (VALIDATION / "zero_inflated" / "observations.tsv").open() as handle:
            selected = [row for row in csv.DictReader(handle, delimiter="\t")
                        if row["dataset"] == case["data_id"]]
        labels = [row["group"] for row in selected]
        data = np.array([[float(row[name]) for name in ("count", "x", "z", "offset")]
                         for row in selected])
        model = fit_zip_random_intercept
    else:
        samples = json.loads((VALIDATION / "zinb2" / "cases.json").read_text(encoding="utf-8"))
        data = np.array(next(sample["observations"] for sample in samples if sample["id"] == case["data_id"]))
        labels = data[:, 1]
        data = data[:, [0, 2, 3, 5, 4]]
        model = fit_zinb2_random_intercept
    ones = np.ones(len(data))
    design = {
        "count_design": np.column_stack((ones, data[:, 1])),
        "inflation_design": np.column_stack((ones, data[:, 2])),
        "offset": data[:, 3],
    }
    if not poisson:
        design["dispersion_design"] = np.column_stack((ones, data[:, 4]))
    fit = model(data[:, 0], labels, **design)
    rows = case["rows"]
    return fit, np.asarray(labels)[rows], {key: values[rows] for key, values in design.items()}


def _check(case, reference):
    fit, groups, design = _model(case)
    assert fit.converged
    fixed = predict_count_mixed(fit, **design)
    labels = groups.astype(object)
    labels[3:] = [None, "unseen"]
    mixed = predict_count_mixed(fit, groups=labels, allow_new_levels=True, **design)
    for scale in ("fixed", "mixed"):
        prediction = fixed if scale == "fixed" else mixed
        for key, values in asdict(prediction).items():
            np.testing.assert_allclose(values, reference[scale][key], atol=2e-6, rtol=4e-6,
                                       err_msg=f"{scale} {key}")
    expected = reference["simulation"]
    samples = simulate_count_mixed(fit, groups, **design, nsim=4800, seed=29417)
    means = samples.mean(axis=0)
    variance = samples.var(axis=0, ddof=1)
    zero = (samples == 0).mean(axis=0)
    r_zero = np.asarray(expected["zero"])
    mean_se = np.sqrt(variance / len(samples) + np.diag(expected["covariance"]) / case["nsim"])
    zero_se = np.sqrt(zero * (1 - zero) / len(samples) + r_zero * (1 - r_zero) / case["nsim"])
    assert (abs(means - expected["means"]) <= 5 * mean_se).all()
    assert (abs(zero - r_zero) <= 5 * zero_se).all()
    centered = samples - means
    product = centered[:, :, None] * centered[:, None, :]
    covariance_se = np.hypot(product.std(axis=0, ddof=1) / np.sqrt(len(samples)),
                             expected["covariance_se"])
    assert (abs(np.cov(samples, rowvar=False) - expected["covariance"]) <= 5 * covariance_se).all()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_predictions_and_simulations_against_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_predictions_and_simulations_against_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=120,
    )
    references = json.loads(process.stdout)["results"]
    for case in CASES:
        _check(case, references[case["id"]])


def test_unknown_groups_and_seeded_simulation():
    fit, groups, design = _model(CASES[0])
    new = ["unseen"] * len(groups)
    with pytest.raises(ValueError, match="absent from the fitted model"):
        predict_count_mixed(fit, groups=new, **design)
    predicted = predict_count_mixed(fit, groups=new, allow_new_levels=True, **design)
    np.testing.assert_array_equal(predicted.link, predict_count_mixed(fit, **design).link)
    first = simulate_count_mixed(fit, new, nsim=10, seed=589, **design)
    np.testing.assert_array_equal(first, simulate_count_mixed(fit, new, nsim=10, seed=589, **design))
    assert first.shape == (10, len(groups))
    assert first.dtype.kind == "i"
