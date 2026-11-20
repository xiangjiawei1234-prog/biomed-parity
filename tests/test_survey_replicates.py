import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.survey_replicates import svrepdesign

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "survey_replicates"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _assert_fields(result, expected):
    for name, reference in expected.items():
        actual = getattr(result, name)
        assert np.shape(actual) == np.shape(reference), name
        np.testing.assert_allclose(actual, np.asarray(reference, dtype=float), rtol=1e-10, atol=1e-12, err_msg=name)


def _check_r_case(case, expected):
    design = svrepdesign(case["repweights"], case.get("weights"), **case.get("options", {}))
    data = np.asarray(case["values"], dtype=float)
    if "subset" in case:
        design = design.subset(case["subset"])
        data = data[case["subset"]]
    _assert_fields(design, {k: v for k, v in expected.items() if k not in ("mean", "total")})

    if "subset" in case:
        with pytest.warns(RuntimeWarning, match="zero weight sums"):
            mean = design.mean(data)
    else:
        mean = design.mean(data, na_rm=case.get("na_rm", False))
    _assert_fields(mean, expected["mean"])
    _assert_fields(design.total(data, na_rm=case.get("na_rm", False)), expected["total"])


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_matches_recorded_r(case):
    _check_r_case(case, EXPECTED[case["name"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r(tmp_path):
    output = tmp_path / "expected.json"
    subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json"), str(output)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    expected = json.loads(output.read_text(encoding="utf-8"))["results"]
    for case in CASES:
        _check_r_case(case, expected[case["name"]])


def test_factor_and_combined_weights_give_same_results_after_row_reordering():
    case = CASES[0]
    raw = svrepdesign(case["repweights"], case["weights"], combined_weights=False)
    order = [3, 7, 2, 5, 1, 4, 6, 0]
    reordered = svrepdesign(raw.analysis_weights[order], raw.sampling_weights[order])
    data = np.asarray(case["values"])
    for method in ("mean", "total"):
        result = getattr(raw, method)(data)
        other = getattr(reordered, method)(data[order])
        for name in ("estimate", "covariance", "replicate_estimates", "design_df"):
            np.testing.assert_allclose(getattr(result, name), getattr(other, name), atol=1e-12)


def test_design_owns_weights_and_subsets_keep_explicit_df():
    weights = np.array([1.0, 2.0])
    repweights = np.array([[2.0, 0], [0, 4.0]])
    rscales = np.array([0.7, 0.9])
    design = svrepdesign(repweights, weights, rscales=rscales, degf=8)
    weights[:] = repweights[:] = rscales[:] = 0
    subset = design.subset([True, False])
    assert subset.total([3]).design_df == 8
    np.testing.assert_array_equal(subset.sampling_weights, [1.0])
    np.testing.assert_array_equal(subset.replicate_weights, [[2.0, 0]])
    np.testing.assert_array_equal(subset.rscales, [0.7, 0.9])


@pytest.mark.parametrize(
    "options, message",
    [
        ({"type": "unknown"}, "type must be"),
        ({"type": "Fay"}, "rho must"),
        ({"type": "Fay", "rho": 1}, "rho must"),
        ({"type": "JKn"}, "rscales is required"),
        ({"weights": [1, 2, 3]}, "one finite value"),
        ({"repweights": [[1, np.nan], [2, 3]]}, "finite matrix"),
        ({"rscales": [1, -1]}, "finite and nonnegative"),
        ({"rscales": [1, 1, 1]}, "one per replicate"),
        ({"type": "other", "scale": -1}, "scale must be nonnegative"),
        ({"fpc": [0.1, 0.2], "fpctype": "fraction"}, "fpc is only available"),
        ({"type": "JK1", "fpc": [0.1, 0.2]}, "fpctype must be"),
        ({"type": "JK1", "fpc": [2, 0], "fpctype": "fraction"}, "one value"),
        ({"type": "bootstrap", "repweights": [[1], [2]]}, "at least two replicates"),
    ],
)
def test_invalid_design(options, message):
    with pytest.raises(ValueError, match=message):
        svrepdesign(**({"repweights": [[2, 0], [0, 2]]} | options))


def test_degenerate_outcome_or_domain():
    design = svrepdesign([[2, 0], [0, 2]])
    with pytest.raises(ValueError, match="use na_rm=True"):
        design.mean([1, np.nan])
    with pytest.raises(ValueError, match="no complete rows"):
        design.total([np.nan, np.nan], na_rm=True)
    with pytest.raises(ValueError, match="one row per sampled unit"):
        design.total([1, 2, 3])
    with pytest.raises(ValueError, match="at least one selected row"):
        design.subset([False, False])
    with pytest.raises(ValueError, match="all replicate weight sums are zero"):
        svrepdesign([[0, 0], [0, 0]]).mean([1, 2])
    with pytest.raises(ValueError, match="sampling weights must sum"):
        svrepdesign([[2, 0], [0, 2]], [1, -1]).mean([1, 2])
