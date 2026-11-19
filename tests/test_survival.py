import json
import os
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.survival import survdiff, survdiff_one_sample
from validation.survival.verify import assert_case, verify

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "survival"
CASES = json.loads((VALIDATION / "cases.json").read_text(encoding="utf-8"))
REFERENCE = {
    case["id"]: case
    for case in json.loads((VALIDATION / "expected.json").read_text(encoding="utf-8"))["cases"]
}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_r_reference(case):
    assert_case(case, REFERENCE[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 for live R")
def test_live_r_reference():
    assert verify(random_count=100)["cases_passed"] == len(CASES) + 100


def test_row_and_factor_order():
    time = np.array([1, 3, 2, 4, 4, 6, 3, 5, 7])
    event = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0])
    group = np.array(["a", "c", "b", "a", "c", "a", "b", "b", "c"])
    original = survdiff(time, event, group, rho=1)
    perm = [8, 3, 6, 7, 5, 0, 2, 4, 1]
    reordered = survdiff(time[perm], event[perm], group[perm], rho=1, group_order=["c", "b", "a"])
    assert reordered.groups == ("c", "b", "a")
    np.testing.assert_allclose(reordered.observed[::-1], original.observed)
    np.testing.assert_allclose(reordered.expected[::-1], original.expected)
    np.testing.assert_allclose(reordered.variance[::-1, ::-1], original.variance)
    assert reordered.statistic == pytest.approx(original.statistic)
    assert reordered.pvalue == pytest.approx(original.pvalue)


@pytest.mark.parametrize(
    ("time", "event", "group", "options", "message"),
    [
        ([], [], [], {}, "time"),
        ([1, np.nan], [1, 0], [1, 2], {}, "finite"),
        ([1, 2], [1], [1, 2], {}, "same length"),
        ([1, 2], [1, 2], [1, 2], {}, "only 0"),
        ([1, 2], [1, 0], [1, 1], {}, "at least two groups"),
        ([1, 2], [1, 0], ["a", None], {}, "finite numeric labels"),
        ([1, 2], [1, 0], [1, 2], {"strata": ["x"]}, "match time"),
        ([1, 2], [1, 0], [1, 2], {"group_order": [2]}, "exactly once"),
        ([1, 2], [1, 0], [1, 2], {"strata_order": [1]}, "requires strata"),
        ([1, 2], [1, 0], [1, 2], {"rho": np.inf}, "finite real"),
        ([1, 2], [1, 0], [1, 2], {"timefix": 1}, "boolean"),
    ],
)
def test_invalid_data(time, event, group, options, message):
    with pytest.raises(ValueError, match=message):
        survdiff(time, event, group, **options)


def test_invalid_one_sample_data():
    with pytest.raises(ValueError, match="same length"):
        survdiff_one_sample([0, 1], [0.8])
    with pytest.raises(ValueError, match="must be in"):
        survdiff_one_sample([0, 1], [0.8, 1.01])
