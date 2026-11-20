import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.meta_influence import meta_hatvalues, meta_influence
from biomed_parity.meta_multilevel import meta_multilevel


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "meta_influence"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _check(case, reference):
    j = np.arange(24)
    group = np.repeat([4, 1, 6, 3, 2, 5], 4)
    x = (j * 7 % 17) / 13
    y = 0.7 - 0.4 * x + 0.6 * np.sin(group * 3) + 0.2 * np.cos(j)
    vi = 0.11 + (j * 3 % 5) / 40
    covariance = 0.2 * np.outer(np.sqrt(vi), np.sqrt(vi)) * (group[:, None] == group)
    np.fill_diagonal(covariance, vi)
    design = np.column_stack((x, group == 4)) if case.get("rare") else x
    fit = meta_multilevel(
        y, covariance, moderators=design, random_intercepts={"group": group} if case["random"] else None,
        test=case["test"],
    )
    diagnosis = meta_influence(
        fit, cluster=group if case["clustered"] else None, reestimate=case["reestimate"],
        cook_coefficients=case.get("cook_coefficients"),
    )
    np.testing.assert_array_equal(diagnosis.labels, reference["labels"])
    refs = {
        "cooks": diagnosis.cooks_distance,
        "dfbetas": diagnosis.dfbetas,
        "deleted_coefficients": diagnosis.deleted_coefficients,
        "residuals": diagnosis.deleted_residuals,
        "residual_se": diagnosis.residual_standard_errors,
        "studentized": diagnosis.studentized_residuals,
        "cluster_chi_squared": diagnosis.cluster_chi_squared,
        "leverage": meta_hatvalues(fit),
        "hat": meta_hatvalues(fit, matrix=True),
    }
    if case["random"]:
        assert diagnosis.variance_names == ("group",)
        refs["deleted_variances"] = diagnosis.deleted_variances[:, 0]
    for name, values in refs.items():
        np.testing.assert_allclose(values, np.asarray(reference[name], dtype=float),
                                   rtol=6e-6, atol=6e-8, err_msg=name)
    estimable = np.isfinite(diagnosis.cluster_chi_squared)
    np.testing.assert_array_equal(
        diagnosis.cluster_df[estimable], np.asarray(reference["cluster_df"], dtype=float)[estimable],
    )
    assert diagnosis.failed_cases == ((4,) if case.get("rare") else ())
    np.testing.assert_allclose(meta_hatvalues(fit).sum(), len(fit.coefficients), atol=2e-14, rtol=0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_multilevel_influence_matches_r_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_multilevel_influence_matches_live_r():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=90,
    )
    results = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, results[case["id"]])


def test_fixed_kernel_context_is_retained_and_labels_remain_aligned():
    effects = np.array([0.1, 0.6, 0.22, 0.4, -0.05, 0.12])
    kernel = np.outer([1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1])
    fit = meta_multilevel(
        effects, np.repeat(0.1, len(effects)), covariance_components={"known": kernel},
        fixed_variances={"known": 0.3},
    )
    diagnosis = meta_influence(fit, cluster=["B", "A", "A", "B", "C", "C"])
    assert diagnosis.labels == ("B", "A", "C")
    np.testing.assert_array_equal(diagnosis.case_index, [0, 1, 1, 0, 2, 2])
    np.testing.assert_array_equal(diagnosis.deleted_variances, np.full((3, 1), 0.3))
    with pytest.raises(ValueError, match="coefficient indices"):
        meta_influence(fit, cook_coefficients=[-1])
