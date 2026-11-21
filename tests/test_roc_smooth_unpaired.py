import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from biomed_parity.roc_smooth_bootstrap import bootstrap_smooth_unpaired_test


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_smooth_unpaired"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]
INDEX_FIELDS = ("bootstrap_indices_1", "bootstrap_indices_2")


def _assert_parity(case, reference):
    data = INPUT["datasets"][case["dataset"]].copy()
    if case.get("same_data"):
        data.update(labels_2=data["labels_1"], scores_2=data["scores_1"])
    options = {key: value for key, value in case.items()
        if key not in ("id", "dataset", "seed", "same_data")}
    options.update({key: reference[key] for key in INDEX_FIELDS})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        result = bootstrap_smooth_unpaired_test(**data, **options)
    omitted = reference["omitted_resamples"]
    assert result.n_resamples == reference["n_resamples"]
    assert result.omitted_resamples == omitted
    assert len(captured) == bool(omitted)
    if omitted:
        assert f"{omitted} bootstrap draws" in str(captured[0].message)
    atol = 3e-11 if case.get("method", "").startswith("logcondens") else 8e-15
    for field in ("auc_1", "auc_2", "difference", "standard_error", "statistic", "p_value"):
        scale = 100 if case.get("percent") and field not in ("statistic", "p_value") else 1
        np.testing.assert_allclose(getattr(result, field), reference[field],
            rtol=4e-14, atol=atol * scale, err_msg=f"{case['id']} {field}")


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
        for key in INDEX_FIELDS:
            np.testing.assert_array_equal(live[key], EXPECTED[case["id"]][key])
        _assert_parity(case, live)


@pytest.mark.parametrize("fixed_sample", [1, 2])
def test_one_fixed_matrix_sets_replicate_count_and_does_not_draw_for_fixed_sample(fixed_sample):
    samples = INPUT["datasets"]["unequal_samples"]
    generated_sample = 3 - fixed_sample
    generated_size = len(samples[f"labels_{generated_sample}"])
    fixed_size = len(samples[f"labels_{fixed_sample}"])
    seed = 731
    count = 5
    fully_fixed = {
        f"bootstrap_indices_{fixed_sample}": np.tile(np.arange(fixed_size), (count, 1)),
        f"bootstrap_indices_{generated_sample}": np.random.default_rng(seed).integers(
            0, generated_size, (count, generated_size)),
    }
    options = dict(method="density", n=31, stratified=False)
    expected = bootstrap_smooth_unpaired_test(**samples, **options, **fully_fixed)
    actual = bootstrap_smooth_unpaired_test(**samples, **options, random_state=seed, n_resamples=13,
        **{f"bootstrap_indices_{fixed_sample}": fully_fixed[f"bootstrap_indices_{fixed_sample}"]})
    assert actual == expected
    assert actual.n_resamples == count


def test_generated_draws_from_identical_inputs_use_independent_samples():
    data = INPUT["datasets"]["unequal_samples"]
    sample = (data["labels_1"], data["scores_1"])
    options = dict(method="density", n=31, n_resamples=16, random_state=731)
    result = bootstrap_smooth_unpaired_test(*sample, *sample, **options)
    assert result == bootstrap_smooth_unpaired_test(*sample, *sample, **options)
    assert result.difference == result.statistic == 0
    assert result.standard_error > 0
    assert result.p_value == 1


def test_rejects_fixed_matrices_with_different_numbers_of_draws():
    data = INPUT["datasets"]["unequal_samples"]
    first = np.tile(np.arange(len(data["labels_1"])), (2, 1))
    second = np.tile(np.arange(len(data["labels_2"])), (3, 1))
    with pytest.raises(ValueError, match="same number of rows"):
        bootstrap_smooth_unpaired_test(**data, method="density", bootstrap_indices_1=first, bootstrap_indices_2=second)
