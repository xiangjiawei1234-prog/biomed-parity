import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_multiclass import multiclass_auc


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_multiclass"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
RTOL = 2e-14
ATOL = 2e-15


def _fit(case):
    data = INPUT["datasets"][case["dataset"]]
    kind = case.get("kind", "single")
    options = {
        key: value for key, value in case.items()
        if key not in ("id", "dataset", "kind", "comparison", "error")
    }
    if kind == "matrix":
        options["score_labels"] = data["columns"]
    return multiclass_auc(data["labels"], np.asarray(data[kind], dtype=float), **options)


def _assert_parity(case, reference):
    if "error" in case:
        assert case["error"] in reference["error"]
        with pytest.raises(ValueError, match="only available for single-score"):
            _fit(case)
        return

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        actual = _fit(case)
    messages = [str(w.message) for w in captured]
    if case.get("comparison") == "upstream_column_error" and "error" in reference:
        assert "Could not find any decision values" in reference["error"]
        assert actual.levels == ("a", "b")
        assert actual.pairs[0].component_aucs == (1.0, 1.0)
        assert actual.auc == 1.0
        assert messages == ["no score columns for selected classes ['c', 'd']"]
        return

    for r_message, py_message in (
        ("You did not provide decision values", "no score columns"),
        ("were not found in 'response'", "ignoring score columns"),
        ("below the diagonal", "below the chance diagonal"),
    ):
        assert any(r_message in text for text in reference["warnings"]) == any(
            py_message in text for text in messages
        )
    assert all(any(known in message for known in (
        "no score columns", "ignoring score columns", "below the chance diagonal",
    )) for message in messages)

    def compare(value, expected, field):
        np.testing.assert_allclose(value, np.asarray(expected, dtype=float), rtol=RTOL,
            atol=ATOL * (100 if case.get("percent") else 1), err_msg=f"{case['id']} {field}")

    compare(actual.auc, reference["auc"], "auc")
    assert actual.levels == tuple(reference["levels"])
    for pair, expected_pair in zip(actual.pairs, reference["pairs"], strict=True):
        assert pair.classes == tuple(expected_pair["classes"])
        assert pair.directions == tuple(expected_pair["directions"])
        compare(pair.auc, expected_pair["auc"], "pair AUC")
        compare(pair.component_aucs, expected_pair["component_aucs"], "binary AUCs")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_proc(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_proc():
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(result.stdout)["results"]
    for case in CASES:
        _assert_parity(case, references[case["id"]])


@pytest.mark.parametrize(
    ("labels", "scores", "options", "message"),
    [
        (["a", "a"], [1, 2], {}, "observed classes"),
        (["a", "b"], [1, 2], {"levels": ["a", "missing"]}, "observed classes"),
        (["a", "b"], [1, 2], {"levels": ["a", "a"]}, "unique nonmissing"),
        (["a", "b"], [1, 2], {"score_labels": ["a"]}, "only used for class-score"),
        (["a", "b"], [1, np.inf], {}, "without infinity"),
        (["a", "b"], [1, np.nan], {}, "usable score"),
        (["a", "b"], [1, np.nan], {"na_rm": False}, "scores in each comparison"),
        (["a", "b", None], [1, 2, 3], {"na_rm": False}, "labels must be present"),
        (["a", "b"], [[1, 2], [2, 1]], {}, "name each column"),
        (["a", "b"], [[1, 2], [2, 1]], {"score_labels": ["a"]}, "one label per score column"),
        (["a", "b"], [[1, 2], [2, 1]], {"score_labels": ["a", "a"]}, "unique nonmissing"),
        (["a", "b"], [[1, 2], [2, 1]], {"score_labels": ["a", "d"]}, "at least two class-score"),
    ],
)
def test_rejects_invalid_or_unestimable_inputs(labels, scores, options, message):
    with pytest.raises(ValueError, match=message):
        multiclass_auc(labels, scores, **options)


def test_reordering_named_columns_keeps_hand_till_mean():
    data = INPUT["datasets"]["unequal_classes"]
    scores = np.asarray(data["matrix"])
    order = [2, 1, 0]
    original = multiclass_auc(data["labels"], scores, score_labels=data["columns"])
    reordered = multiclass_auc(data["labels"], scores[:, order],
        score_labels=[data["columns"][i] for i in order])
    assert reordered.levels == ("c", "a", "b")
    assert reordered.auc == pytest.approx(original.auc, rel=RTOL, abs=ATOL)
