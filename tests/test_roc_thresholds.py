"""Compare empirical ROC operating points and bootstrap coordinate intervals with pROC."""

import csv
import collections
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_thresholds import bootstrap_roc_coords, roc_coords


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "roc_thresholds"

# The only observed difference is the last bit of the type-7 bootstrap
# quantile interpolation (max relative deviation 2.0e-16, max absolute
# deviation 5.6e-17 over the snapshot).
RTOL = 2e-14
ATOL = 2e-15


def read_tsv(lines):
    return list(csv.DictReader((line for line in lines if not line.startswith("#")), delimiter="\t"))


def fixture(name):
    return read_tsv((FIXTURES / f"{name}.tsv").read_text().splitlines())


OBSERVATIONS = fixture("observations")
LABELS = [int(row["label"]) for row in OBSERVATIONS]
SCORE_NAMES = [name for name in OBSERVATIONS[0] if name not in ("subject", "label")]
SCORES = {
    name: [float(row[name]) for row in OBSERVATIONS] for name in SCORE_NAMES
}
CASES = fixture("cases")
EXPECTED = fixture("expected")
EXPECTED_BY_ID = collections.defaultdict(list)
for row in EXPECTED:
    EXPECTED_BY_ID[row["id"]].append(row)
DRAW_NAMES = ["draws_main", "draws_perfect", "draws_policy", "draws_random", "draws_nonstrat"]
DRAWS = {
    name: np.loadtxt(FIXTURES / f"{name}.tsv", delimiter="\t", skiprows=1, dtype=int)
    for name in DRAW_NAMES
}
COORD_CASES = [case for case in CASES if case["kind"] == "coords"]
BOOT_CASES = [case for case in CASES if case["kind"] in ("ci", "ci_error")]


def parse_x(value):
    if value.startswith("num:"):
        return [float(item) for item in value[4:].split(",")]
    return value


def parse_ret(value):
    if not value:
        return None
    if value == "all":
        return "all"
    return tuple(value.split(","))


def case_kwargs(case):
    return {
        "x": parse_x(case["x"]),
        "input": case["input"],
        "direction": case["direction"],
        "best_method": case["best_method"],
        "best_weights": tuple(float(item) for item in case["best_weights"].split(",")),
        "partial_auc": (
            None
            if not case["partial_auc"]
            else tuple(float(item) for item in case["partial_auc"].split(","))
        ),
        "focus": case["focus"],
        "percent": case["percent"] == "TRUE",
    }


def case_ret(case):
    return parse_ret(case["ret"])


def expected_number(text):
    if text == "NA":
        return np.nan
    return float(text)


def assert_value(actual, text, label):
    expected = expected_number(text)
    if np.isnan(expected):
        assert np.isnan(actual), f"{label}: expected NA, got {actual}"
    elif np.isinf(expected):
        assert actual == expected, f"{label}: expected {expected}, got {actual}"
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=label)


def expected_matrix(rows, kind):
    mapping = {
        (int(row["row"]), row["field"]): row["value"]
        for row in rows
        if row["kind"] == kind
    }
    if not mapping:
        return mapping, []
    last_row = max(row for row, _ in mapping)
    fields = [field for (row, field) in mapping if row == 1]
    assert len(fields) == len(set(fields))
    return mapping, fields


def assert_matrix(actual, rows, kind, label):
    mapping, fields = expected_matrix(rows, kind)
    actual = np.atleast_2d(actual)
    expected_rows = max(row for row, _ in mapping)
    assert actual.shape == (expected_rows, len(fields)), (
        f"{label}: got {actual.shape}, expected {(expected_rows, len(fields))}"
    )
    for i in range(expected_rows):
        for j, field in enumerate(fields):
            assert_value(actual[i, j], mapping[(i + 1, field)], f"{label}[{i + 1},{field}]")


def expected_summary(rows):
    return {row["field"]: row["value"] for row in rows if row["kind"] == "summary"}


@pytest.mark.parametrize("case", COORD_CASES, ids=lambda case: case["id"])
def test_roc_coords_matches_proc(case):
    kwargs = case_kwargs(case)
    ret = case_ret(case)
    if ret is not None:
        kwargs["ret"] = ret
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = roc_coords(LABELS, SCORES[case["score"]], **kwargs)
    assert not caught, [str(warning.message) for warning in caught]
    assert_matrix(
        actual, EXPECTED_BY_ID[case["id"]], "coord", case["id"]
    )


@pytest.mark.parametrize("case", BOOT_CASES, ids=lambda case: case["id"])
def test_bootstrap_roc_coords_matches_proc(case):
    kwargs = case_kwargs(case)
    kwargs.update(
        best_policy=case["best_policy"],
        stratified=case["stratified"] == "TRUE",
        confidence_level=float(case["confidence_level"]),
        bootstrap_indices=DRAWS[case["draws"]],
        random_state=0,
    )
    ret = case_ret(case)
    if ret is not None:
        kwargs["ret"] = ret
    rows = EXPECTED_BY_ID[case["id"]]
    if case["kind"] == "ci_error":
        assert any(row["kind"] == "error" for row in rows)
        with pytest.raises(ValueError, match="best threshold"):
            bootstrap_roc_coords(LABELS, SCORES[case["score"]], **kwargs)
        return
    summary = expected_summary(rows)
    omitted = int(float(summary["omitted"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bootstrap_roc_coords(LABELS, SCORES[case["score"]], **kwargs)
    if omitted:
        assert any("omitted" in str(warning.message) for warning in caught)
    else:
        assert not caught, [str(warning.message) for warning in caught]

    assert_matrix(result.observed, rows, "point", case["id"] + " observed")
    _, fields = expected_matrix(rows, "point")
    intervals = {
        (int(row["row"]), row["field"], row["quantile"]): row["value"]
        for row in rows
        if row["kind"] == "interval"
    }
    points = max(point for point, _, _ in intervals)
    for point in range(1, points + 1):
        for j, field in enumerate(fields):
            for q, quantile in enumerate(("lower", "median", "upper")):
                assert_value(
                    result.intervals[point - 1, j, q],
                    intervals[(point, field, quantile)],
                    f"{case['id']}.interval[{point},{field},{quantile}]",
                )
    counts = {
        (int(row["row"]), row["field"]): row["value"]
        for row in rows
        if row["kind"] == "count"
    }
    assert counts
    for (point, field), value in counts.items():
        assert_value(
            result.counts[point - 1, fields.index(field)],
            value,
            f"{case['id']}.count[{point},{field}]",
        )
    assert_value(result.n_resamples, summary["n_resamples"], case["id"] + ".n_resamples")
    assert_value(result.omitted_resamples, summary["omitted"], case["id"] + ".omitted")


def resampled_best(labels, scores, draw):
    return roc_coords(
        np.asarray(labels)[draw], np.asarray(scores)[draw], x="best",
        ret=("threshold", "specificity", "sensitivity"),
    )


def policy_draws():
    """Split the stored ambiguous matrix into one clean and one ambiguous draw."""
    labels = np.asarray(LABELS)
    scores = np.asarray(SCORES["score_tiebest"])
    clean = ambiguous = None
    for draw in DRAWS["draws_policy"]:
        best = resampled_best(labels, scores, draw)
        if len(best) == 1:
            clean = draw
        elif len(best) > 1:
            ambiguous = draw
    assert clean is not None and ambiguous is not None
    return np.vstack((clean, ambiguous))


def test_best_policy_stop_omit_and_random_use_documented_semantics():
    matrix = policy_draws()
    labels, scores = LABELS, SCORES["score_tiebest"]
    clean = resampled_best(np.asarray(labels), np.asarray(scores), matrix[0])

    with pytest.raises(ValueError, match="best threshold"):
        bootstrap_roc_coords(
            labels, scores, x="best", best_policy="stop", bootstrap_indices=matrix
        )
    with pytest.warns(RuntimeWarning, match="omitted"):
        omitted_result = bootstrap_roc_coords(
            labels, scores, x="best", best_policy="omit", bootstrap_indices=matrix
        )
    assert omitted_result.n_resamples == 2
    assert omitted_result.omitted_resamples == 1
    np.testing.assert_array_equal(omitted_result.counts, [[1] * 3])
    np.testing.assert_allclose(
        omitted_result.intervals[0, :, 1], clean[0], rtol=0, atol=0
    )
    for j in range(3):
        np.testing.assert_array_equal(
            omitted_result.intervals[0, j, :], np.repeat(clean[0, j], 3)
        )

    # Two ambiguous draws give the random policy two independent row choices.
    two_ambiguous = np.vstack((matrix[1], matrix[1]))
    candidates = resampled_best(np.asarray(labels), np.asarray(scores), matrix[1])[:, 1]
    medians = set()
    for seed in range(24):
        result = bootstrap_roc_coords(
            labels, scores, x="best", ret="specificity", best_policy="random",
            bootstrap_indices=two_ambiguous, random_state=seed,
        )
        assert result.omitted_resamples == 0
        np.testing.assert_array_equal(result.counts, [[2]])
        median = result.intervals[0, 0, 1]
        assert any(np.isclose(median, (a + b) / 2) for a in candidates for b in candidates)
        medians.add(round(float(median), 12))
    assert len(medians) > 1


def test_roc_coords_input_validation():
    labels, scores = LABELS, SCORES["score_main"]
    with pytest.raises(ValueError, match="outside"):
        roc_coords(labels, scores, x=[-0.1], input="specificity")
    with pytest.raises(ValueError, match="direction"):
        roc_coords(labels, scores, direction="sideways")
    with pytest.raises(ValueError, match="best_method"):
        roc_coords(labels, scores, x="best", best_method="topleft")
    with pytest.raises(ValueError, match="best_weights"):
        roc_coords(labels, scores, x="best", best_weights=(1, 1))
    with pytest.raises(ValueError, match="unsupported ROC coordinate"):
        roc_coords(labels, scores, ret=("auc",))


def test_bootstrap_roc_coords_input_validation():
    labels, scores = LABELS, SCORES["score_main"]
    draws = DRAWS["draws_main"]
    with pytest.raises(ValueError, match="x='best'"):
        bootstrap_roc_coords(labels, scores, x="all", bootstrap_indices=draws)
    with pytest.raises(ValueError, match="threshold intervals"):
        bootstrap_roc_coords(
            labels, scores, x=[0.5], input="specificity", ret=("threshold",),
            bootstrap_indices=draws,
        )
    with pytest.raises(ValueError, match="best_policy"):
        bootstrap_roc_coords(
            labels, scores, x="best", best_policy="guess", bootstrap_indices=draws
        )


@pytest.mark.skipif(
    os.environ.get("BIOMED_R_PARITY") != "1", reason="set BIOMED_R_PARITY=1 to run pROC"
)
def test_regenerate_roc_thresholds_oracle():
    env = os.environ.copy()
    if (ROOT / ".r-lib").is_dir():
        env.setdefault("R_LIBS_USER", str(ROOT / ".r-lib"))
    command = ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES)]
    draws_text = subprocess.run(
        [*command, "--draws"], cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    ).stdout
    blocks = {}
    current = None
    for line in draws_text.splitlines():
        if line.startswith("# "):
            current = line[2:].strip()
            blocks[current] = []
        elif line.strip():
            blocks[current].append([int(item) for item in line.split("\t")])
    assert set(blocks) == set(DRAWS)
    for name, rows in blocks.items():
        np.testing.assert_array_equal(np.asarray(rows), DRAWS[name])

    expected_text = subprocess.run(
        command, cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    ).stdout
    live_rows = read_tsv(expected_text.splitlines())
    live_by_id = collections.defaultdict(list)
    for row in live_rows:
        live_by_id[row["id"]].append(row)
    assert live_by_id.keys() == EXPECTED_BY_ID.keys()
    for case_id, rows in live_by_id.items():
        assert len(rows) == len(EXPECTED_BY_ID[case_id]), case_id
        for live, stored in zip(rows, EXPECTED_BY_ID[case_id]):
            assert live["kind"] == stored["kind"], case_id
            assert live["row"] == stored["row"], case_id
            assert live["field"] == stored["field"], case_id
            assert live["quantile"] == stored["quantile"], case_id
            if live["kind"] == "error":
                assert live["value"] == stored["value"], case_id
                continue
            live_value, stored_value = expected_number(live["value"]), expected_number(stored["value"])
            if np.isnan(live_value):
                assert np.isnan(stored_value), case_id
            else:
                np.testing.assert_allclose(
                    live_value, stored_value, rtol=0, atol=0, err_msg=case_id
                )
