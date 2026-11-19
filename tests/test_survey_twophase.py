import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.survey_twophase import survey_twophase

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "survey_twophase"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _arguments(case):
    return {key: value for key, value in case.items() if key not in ("name", "audit_unequal_phase1")}


def _check_r_case(case, expected):
    name = case["name"]
    for mode in ("mean", "total"):
        fit = survey_twophase(**_arguments(case), statistic=mode)
        for field, value in expected[name][mode].items():
            actual = getattr(fit, field)
            np.testing.assert_array_equal(np.shape(actual), np.shape(value))
            np.testing.assert_allclose(actual, value, rtol=1e-9, atol=1e-11, err_msg=f"{name}/{mode}/{field}")
        for field in ("weights", "phase1_probabilities", "phase2_probabilities"):
            np.testing.assert_allclose(getattr(fit, field), expected[name][field], rtol=1e-10)
        assert fit.categories == (tuple(case["categories"]) if "categories" in case else None)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_matches_recorded_r(case):
    _check_r_case(case, EXPECTED)


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r(tmp_path):
    output = tmp_path / "expected.json"
    subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json"), str(output)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    expected = json.loads(output.read_text(encoding="utf-8"))
    for case in CASES:
        _check_r_case(case, expected)


def _probability(n, population, same_stratum, same_unit, other_n=0, other_population=1):
    if same_unit:
        return n / population
    if same_stratum:
        return n * (n - 1) / (population * (population - 1))
    return (n / population) * (other_n / other_population)


def test_phase_one_covariance_uses_full_frame_counts_and_is_row_order_invariant():
    # Recalculate the two-stage pair probabilities directly on this small case.
    case = next(item for item in CASES if item.get("audit_unequal_phase1"))
    fit = survey_twophase(**_arguments(case), statistic="total")
    p1_labels = case["phase1_strata"]
    p2_labels = case["phase2_strata"]
    clusters = list(zip(p2_labels, case["phase2_psu"]))
    p1_sizes = Counter(p1_labels)
    p1_pop = case["phase1_population"]
    sampled = np.flatnonzero(case["phase2"])
    p2_units = Counter(clusters)
    p2_sampled_units = Counter(clusters[i] for i in sampled)
    p2_clusters = Counter(stratum for stratum, _ in p2_units)
    p2_sampled_clusters = Counter(stratum for stratum, _ in p2_sampled_units)

    pair_factors = np.zeros((sampled.size, sampled.size))
    phase2_factors = np.zeros_like(pair_factors)
    for row, i in enumerate(sampled):
        for col, j in enumerate(sampled):
            s_i, s_j = p1_labels[i], p1_labels[j]
            first_joint = _probability(
                p1_sizes[s_i], p1_pop[s_i], s_i == s_j, i == j, p1_sizes[s_j], p1_pop[s_j],
            )
            g_i, g_j = p2_labels[i], p2_labels[j]
            c_i, c_j = clusters[i], clusters[j]
            second_psu_joint = _probability(
                p2_sampled_clusters[g_i], p2_clusters[g_i], g_i == g_j, c_i == c_j,
                p2_sampled_clusters[g_j], p2_clusters[g_j],
            )
            second_unit_joint = _probability(
                p2_sampled_units[c_i], p2_units[c_i], c_i == c_j, i == j,
                p2_sampled_units[c_j], p2_units[c_j],
            )
            second_joint = second_psu_joint * second_unit_joint
            second_ratio = fit.phase2_probabilities[row] * fit.phase2_probabilities[col] / second_joint
            first_ratio = fit.phase1_probabilities[row] * fit.phase1_probabilities[col] / first_joint
            pair_factors[row, col] = 1 - first_ratio * second_ratio
            phase2_factors[row, col] = 1 - second_ratio

    weighted = np.asarray(case["values"])[sampled] * fit.weights[:, None]
    np.testing.assert_allclose(fit.covariance, weighted.T @ pair_factors @ weighted, rtol=1e-10)
    np.testing.assert_allclose(fit.phase2_covariance, weighted.T @ phase2_factors @ weighted, rtol=1e-10)
    np.testing.assert_allclose(fit.covariance, fit.covariance.T, rtol=1e-10)
    original_r = np.asarray(EXPECTED[case["name"]]["original_row_order"]["total"]["covariance"])
    assert not np.allclose(original_r, original_r.T, rtol=1e-10)

    shuffled = _arguments(case)
    order = np.random.default_rng(20011).permutation(len(case["phase2"]))
    for field in ("values", "phase2", "phase1_strata", "phase2_strata", "phase2_psu"):
        shuffled[field] = np.asarray(case[field])[order]
    other = survey_twophase(**shuffled, statistic="total")
    np.testing.assert_allclose(other.estimate, fit.estimate, rtol=1e-10)
    np.testing.assert_allclose(other.covariance, fit.covariance, rtol=1e-10)


def test_numeric_and_category_inputs_can_omit_unmeasured_rows():
    selected = [True, False, True, False]
    full = survey_twophase([3, None, 9, None], selected)
    observed = survey_twophase([3, 9], selected)
    np.testing.assert_array_equal(full.estimate, observed.estimate)
    np.testing.assert_array_equal(full.covariance, observed.covariance)

    categories = [9, 3, 7]
    outcome = survey_twophase([3, None, 9, None], selected, categories=categories)
    assert outcome.categories == tuple(categories)
    np.testing.assert_array_equal(outcome.estimate, [0.5, 0.5, 0])


@pytest.mark.parametrize(
    "replacement, message",
    [
        ({"phase2": [1, 2, 0, 0]}, "selection mask"),
        ({"phase2": [0, 0, 0, 0]}, "selection mask"),
        ({"statistic": "variance"}, "must be"),
        ({"phase1_population": 3}, "at least"),
        ({"phase1_population": 4.1}, "integers"),
        ({"phase1_strata": ["a"] * 4}, "supply phase1_population"),
        ({"phase2_strata": ["a", "b", "a", "b"]}, "must contain a sampled unit"),
        ({"phase2_psu": [1, 2]}, "one label"),
        ({"values": [np.nan, 6, 9, 3]}, "must be finite"),
        ({"values": [3, 4, 5]}, "one row"),
        ({"categories": [3, 9, np.nan]}, "unique finite"),
    ],
)
def test_invalid_inputs(replacement, message):
    standard = {"values": [3, 6, 9, 3], "phase2": [1, 0, 1, 0]}
    with pytest.raises(ValueError, match=message):
        survey_twophase(**(standard | replacement))
