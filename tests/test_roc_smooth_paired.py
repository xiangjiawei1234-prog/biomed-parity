import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_smooth_bootstrap import bootstrap_smooth_paired_test


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_smooth_paired"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _fit(case, reference):
    data = INPUT["datasets"][case["dataset"]].copy()
    if case.get("identical_scores", False):
        data["scores_2"] = data["scores_1"]
    options = {name: value for name, value in case.items()
        if name not in ("id", "dataset", "seed", "reuse_auc", "identical_scores", "append_failed_pair")}
    return bootstrap_smooth_paired_test(**data, **options, bootstrap_indices=reference["bootstrap_indices"])


def _assert_parity(case, reference):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        result = _fit(case, reference)
    omitted = reference["omitted_resamples"]
    assert result.n_resamples == reference["n_resamples"]
    assert result.omitted_resamples == omitted
    assert len(captured) == bool(omitted)
    if omitted:
        assert f"{omitted} bootstrap draws" in str(captured[0].message)
    for field in ("auc_1", "auc_2", "difference", "standard_error", "statistic", "p_value"):
        scale = 100 if case.get("percent") and field not in ("statistic", "p_value") else 1
        np.testing.assert_allclose(getattr(result, field), np.asarray(reference[field], dtype=float),
            rtol=4e-14, atol=8e-15 * scale, err_msg=f"{case['id']} {field}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_p_roc(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_p_roc():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    results = json.loads(process.stdout)["results"]
    for case in CASES:
        live = results[case["id"]]
        np.testing.assert_array_equal(live["bootstrap_indices"], EXPECTED[case["id"]]["bootstrap_indices"])
        _assert_parity(case, live)


@pytest.mark.parametrize("valid_draw_count", [0, 1])
def test_fewer_than_two_usable_draws_leave_observed_aucs(valid_draw_count):
    data = INPUT["datasets"]["zero_controls"]
    labels = np.asarray(data["labels"])
    failed = np.r_[np.repeat(0, (labels == 0).sum()), np.repeat(1, (labels == 1).sum())]
    draws = [np.arange(labels.size)] * valid_draw_count + [failed] * (2 - valid_draw_count)
    with pytest.warns(RuntimeWarning, match=f"{2 - valid_draw_count} bootstrap draws"):
        result = bootstrap_smooth_paired_test(**data, method="fitdistr",
            control_distribution="exponential", case_distribution="exponential", bootstrap_indices=draws)
    assert np.isfinite([result.auc_1, result.auc_2, result.difference]).all()
    assert np.isnan([result.standard_error, result.statistic, result.p_value]).all()


def test_rejects_misaligned_scores():
    data = INPUT["datasets"]["paired_scores"].copy()
    data["scores_2"] = data["scores_2"][:-1]
    with pytest.raises(ValueError, match="arrays of equal size"):
        bootstrap_smooth_paired_test(**data)
