import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.survey_jackknife import as_svrepdesign
from biomed_parity.survey_replicates import SurveyReplicateDesign


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "survey_jackknife"
FIXTURE = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _assert_numeric_fields(result, expected):
    for name, target in expected.items():
        value = getattr(result, name)
        assert np.shape(value) == np.shape(target), name
        np.testing.assert_allclose(value, target, rtol=1e-10, atol=1e-11, err_msg=name)


def _check_r_case(case, references):
    expected = references[case["name"]]
    settings = FIXTURE["designs"][case["design"]] | case.get("options", {})
    if "error" in case:
        assert case["error"] in expected["error"]
        message = "only one noncertainty PSU" if case["name"] == "jkn_lonely_fail" else "no jackknife replicates remain"
        with pytest.raises(ValueError, match=message):
            as_svrepdesign(FIXTURE["weights"], **settings)
        return

    design = as_svrepdesign(FIXTURE["weights"], **settings)
    assert isinstance(design, SurveyReplicateDesign)
    assert design.type == expected["type"]
    assert design.combined_weights is expected["combined_weights"]
    assert design.mse is expected["mse"]
    _assert_numeric_fields(design, {k: v for k, v in expected.items() if k not in (
        "type", "combined_weights", "mse", "mean", "total",
    )})
    _assert_numeric_fields(design.mean(FIXTURE["outcomes"]), expected["mean"])
    _assert_numeric_fields(design.total(FIXTURE["outcomes"]), expected["total"])


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_matches_recorded_r(case):
    _check_r_case(case, EXPECTED)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    expected = json.loads(process.stdout)["results"]
    for case in FIXTURE["cases"]:
        _check_r_case(case, expected)


def test_stratum_mapping_and_statistics_are_row_order_invariant():
    order = [7, 8, 9, 0, 4, 1, 5, 3, 2, 6]
    outcomes = np.asarray(FIXTURE["outcomes"])
    weights = np.asarray(FIXTURE["weights"])
    design = FIXTURE["designs"]["nested_hospital"]
    fpc = {"urban": 0.2, "rural": 0.4}
    original = as_svrepdesign(weights, **design, fpc=fpc)
    reordered = as_svrepdesign(weights[order], **{k: np.asarray(v)[order] for k, v in design.items()}, fpc=fpc)
    for method in ("mean", "total"):
        expected = getattr(original, method)(outcomes)
        result = getattr(reordered, method)(outcomes[order])
        np.testing.assert_allclose(result.estimate, expected.estimate)
        np.testing.assert_allclose(result.covariance, expected.covariance)
        assert result.design_df == expected.design_df


@pytest.mark.parametrize("options, message", [
    ({"weights": [1, -1, 2, 1]}, "nonnegative"),
    ({"psu": ["A"]}, "one label per sampled row"),
    ({"psu": ["A"] * 4}, "at least two sampled PSUs"),
    ({"type": "JKn"}, "JKn requires strata"),
    ({"type": "JK1", "strata": [1, 1, 2, 2]}, "JK1 requires an unstratified design"),
    ({"type": "BRR"}, "type must be"),
    ({"lonely_psu": "unknown"}, "lonely_psu must be"),
    ({"fpc": 1.1}, r"must be in \[0, 1\]"),
    ({"fpc": 3, "fpctype": "population"}, "at least the number"),
    ({"fpc": [0.1, 0.2, 0.1, 0.1]}, "constant within each stratum"),
    ({"fpc": {"A": 0.4}, "strata": ["A", "A", "B", "B"]}, "a value for each stratum"),
    ({"fpc": 1}, "no JK1 replicates remain"),
])
def test_invalid_input(options, message):
    with pytest.raises(ValueError, match=message):
        as_svrepdesign(**({"weights": [1, 2, 3, 4]} | options))
