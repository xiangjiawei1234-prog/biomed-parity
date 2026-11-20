import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_smooth_bootstrap import bootstrap_smooth_auc_ci, bootstrap_smooth_sensitivity_ci


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_smooth_bootstrap"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
LINKED = json.loads((FIXTURES.parent / "roc_smoothing" / "cases.json").read_text(encoding="utf-8"))
DATASETS = INPUT["datasets"] | LINKED["datasets"]
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _assert_parity(case, reference):
    sensitivity = case.get("kind") == "sensitivity"
    function = bootstrap_smooth_sensitivity_ci if sensitivity else bootstrap_smooth_auc_ci
    settings = {
        key: value for key, value in case.items()
        if key not in ("id", "seed", "dataset", "kind", "append_draw")
    }
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        actual = function(**DATASETS[case["dataset"]], **settings,
            bootstrap_indices=reference["bootstrap_indices"])
    expected_messages = []
    if reference["observed"] == [None]:
        expected_messages.append("partial AUC correction is undefined below the chance diagonal")
    if reference["omitted_resamples"]:
        expected_messages.append(f"{reference['omitted_resamples']} bootstrap draws")
    assert len(captured) == len(expected_messages)
    assert all(message in str(warning.message) for message, warning in zip(expected_messages, captured, strict=True))

    assert actual.n_resamples == reference["n_resamples"]
    assert actual.omitted_resamples == reference["omitted_resamples"]
    if sensitivity:
        np.testing.assert_array_equal(actual.specificity, np.atleast_1d(case["specificity"]))
        point, interval = actual.sensitivity, actual.intervals
    else:
        lower, upper = actual.confidence_interval
        point, interval = [actual.auc], [[lower, actual.median, upper]]
    scale = 100 if case.get("percent") else 1
    for value, expected in ((point, reference["observed"]), (interval, reference["intervals"])):
        np.testing.assert_allclose(value, np.asarray(expected, dtype=float),
            rtol=2e-14, atol=3e-15 * scale, err_msg=case["id"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_p_roc(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_p_roc():
    result = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(result.stdout)["results"]
    for case in CASES:
        original = EXPECTED[case["id"]]
        live = references[case["id"]]
        np.testing.assert_array_equal(original["bootstrap_indices"], live["bootstrap_indices"])
        _assert_parity(case, live)


@pytest.mark.parametrize("stratified", [True, False])
def test_seed_uses_subject_indices(stratified):
    data = DATASETS["continuous"]
    labels = np.asarray(data["labels"])
    controls, cases = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(335)
    count = 9
    draws = [
        np.r_[rng.choice(controls, size=len(controls), replace=True), rng.choice(cases, size=len(cases), replace=True)]
        if stratified else rng.integers(0, labels.size, labels.size)
        for _ in range(count)
    ]
    settings = dict(method="density", direction="auto", n=37, stratified=stratified)
    generated = bootstrap_smooth_auc_ci(**data, **settings, n_resamples=count, random_state=335)
    fixed = bootstrap_smooth_auc_ci(**data, **settings, n_resamples=1, bootstrap_indices=draws)
    assert generated == fixed
    assert fixed.n_resamples == count


def test_all_draws_omitted_leave_point_estimates_and_missing_bounds():
    data = DATASETS["continuous"]
    indices = [[0] * 10 + [10] * 7] * 2
    with pytest.warns(RuntimeWarning, match="2 bootstrap draws"):
        area = bootstrap_smooth_auc_ci(**data, bootstrap_indices=indices)
    with pytest.warns(RuntimeWarning, match="2 bootstrap draws"):
        sensitivity = bootstrap_smooth_sensitivity_ci(**data, specificity=0.5, bootstrap_indices=indices)
    assert np.isfinite(area.auc)
    assert np.isnan([*area.confidence_interval, area.median]).all()
    assert np.isfinite(sensitivity.sensitivity).all()
    np.testing.assert_array_equal(sensitivity.intervals, [[np.nan] * 3])


def test_single_usable_draw_keeps_pointwise_interval_shape():
    data = DATASETS["continuous"]
    indices = [np.arange(17), [0] * 10 + [10] * 7]
    with pytest.warns(RuntimeWarning, match="1 bootstrap draws"):
        result = bootstrap_smooth_sensitivity_ci(**data, specificity=[0.3, 0.5], bootstrap_indices=indices)
    np.testing.assert_array_equal(result.intervals, np.repeat(result.sensitivity[:, None], 3, axis=1))


@pytest.mark.parametrize("options,message", [
    ({"confidence_level": 1.01}, "confidence_level"),
    ({"n_resamples": 1}, "n_resamples"),
    ({"bootstrap_indices": [[1] * 17] * 2}, "preserve the original class counts"),
    ({"bootstrap_indices": [[-1] * 17] * 2, "stratified": False}, "in-range"),
])
def test_invalid_bootstrap_options(options, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_smooth_auc_ci(**DATASETS["continuous"], **options)


@pytest.mark.parametrize("specificity", [[], [-0.01], [1.01], [[0.4]], [np.nan]])
def test_invalid_specificity_values(specificity):
    with pytest.raises(ValueError, match="specificity must contain finite values"):
        bootstrap_smooth_sensitivity_ci(**DATASETS["continuous"], specificity=specificity)
