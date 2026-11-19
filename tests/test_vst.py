"""Compare the local VST with DESeq2 on a common dispersion trend."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.vst import DispersionGrid, local_vst, local_vst_grid


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "vst"
SPEC = json.loads((FIXTURES / "cases.json").read_text())
REFERENCE = json.loads((FIXTURES / "expected.json").read_text())["results"]


def _inputs(case, reference):
    raw = np.asarray(SPEC["counts"][case["counts"]])
    curve = case["curve"]
    factors = case.get("size_factors")
    if curve["type"] == "sampled":
        trend = DispersionGrid(curve["mean"], curve["dispersion"])
    elif curve["type"] == "fitted":
        trend = DispersionGrid(reference["means"], reference["dispersions"])
        factors = reference["size_factors"]
        np.testing.assert_allclose(
            local_vst_grid(raw / factors), reference["means"], rtol=1.5e-14, atol=1e-14,
        )
    else:
        def trend(means):
            return (
                curve["base"] + curve["slope"] / (means + curve["shift"])
                + curve["hump"] * np.exp(-((np.log1p(means) - 2.5) / 1.2)**2)
            )
    return raw, trend, {"size_factors": factors,
                        "normalization_factors": case.get("normalization_factors")}


def _compare(case, reference):
    raw, trend, options = _inputs(case, reference)
    observed = local_vst(raw, trend, **options)
    assert observed.shape == raw.shape
    np.testing.assert_allclose(observed, reference["transform"], atol=1e-10, rtol=2e-12)


@pytest.mark.parametrize("case", SPEC["cases"], ids=lambda case: case["id"])
def test_local_vst_matches_deseq2(case):
    _compare(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1")
def test_live_deseq2_reference():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=60,
    )
    reference = json.loads(process.stdout)["results"]
    for case in SPEC["cases"]:
        _compare(case, reference[case["id"]])


def test_explicitly_normalized_and_raw_features_agree():
    case = next(case for case in SPEC["cases"] if case["id"] == "per-feature-factors")
    raw, trend, options = _inputs(case, REFERENCE[case["id"]])
    normalized = raw / options["normalization_factors"]
    direct = local_vst(normalized, trend, normalized=True, **options)
    implicit = local_vst(raw, trend, **options)
    np.testing.assert_array_equal(direct, implicit)


def test_default_unit_factors_and_sampled_callable_agree():
    counts = np.asarray(SPEC["counts"]["small"])
    grid = local_vst_grid(counts)
    trend = DispersionGrid(grid, np.full_like(grid, 0.2))
    expected = local_vst(counts, lambda q: 0.2, size_factors=np.ones(counts.shape[1]))
    np.testing.assert_array_equal(local_vst(counts, trend), expected)
    assert len(grid) == 999
    assert grid[-1] == counts.max()


def test_column_and_feature_permutations_preserve_values():
    case = next(case for case in SPEC["cases"] if case["id"] == "per-feature-factors")
    raw, trend, options = _inputs(case, REFERENCE[case["id"]])
    rows = np.random.default_rng(70105).permutation(raw.shape[0])
    columns = np.random.default_rng(521).permutation(raw.shape[1])
    normalized_order = np.ix_(rows, columns)
    expected = local_vst(raw, trend, **options)
    observed = local_vst(
        raw[normalized_order], trend,
        normalization_factors=np.asarray(options["normalization_factors"])[normalized_order],
    )
    np.testing.assert_allclose(observed, expected[normalized_order], atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize(
    ("counts", "options", "message"),
    [
        ([[0,0],[0,0]], {}, "positive normalized count"),
        ([[1,3],[3,1]], {}, "quantiles"),
        ([[0,0],[1,-1]], {}, "nonnegative"),
        ([[0,0],[3,1]], {"size_factors": [1,0]}, "positive"),
        ([[0,0],[3,1]], {"normalization_factors": [[1,1]]}, "match counts"),
        ([[0,0],[3,1]], {"normalized": "yes"}, "boolean"),
    ],
)
def test_rejects_invalid_matrix_and_undefined_anchors(counts, options, message):
    with pytest.raises(ValueError, match=message):
        local_vst(counts, lambda q: 0.2, **options)


def test_rejects_negative_dispersions_and_uncovered_grid():
    counts = np.asarray(SPEC["counts"]["small"])
    with pytest.raises(ValueError, match="entire integration grid"):
        local_vst(counts, DispersionGrid([0,1], [0.2,0.1]))
    with pytest.raises(ValueError, match="nonnegative"):
        local_vst(counts, lambda q: -0.2)
    with pytest.raises(ValueError, match="strictly increasing"):
        DispersionGrid([0,1,1], [0.2,0.1,0.1])
