"""Validate two-level Gaussian imputation against the mice 2l.pan and 2l.norm kernels."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from biomed_parity.chained_imputation import mice_impute

from biomed_parity.multilevel_imputation import (
    TwoLevelOptions,
    _design,
    impute_two_level_normal,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "multilevel_imputation"


def _json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


STRUCTURE_SPEC = _json("cases.json")
STRUCTURE_CASES = STRUCTURE_SPEC["cases"]
STRUCTURE_EXPECTED = _json("expected.json")["results"]
DISTRIBUTION_SPEC = _json("distribution_cases.json")
DISTRIBUTION_CASES = DISTRIBUTION_SPEC["cases"]
DISTRIBUTION_EXPECTED = _json("distribution_expected.json")["results"]

STRUCTURE_IDS = [case["id"] for case in STRUCTURE_CASES]
DISTRIBUTION_IDS = [case["id"] for case in DISTRIBUTION_CASES]


def _response_predictors(spec, case):
    dataset = spec["datasets"][case["dataset"]]
    columns = dataset["columns"]
    matrix = np.array(
        [[np.nan if value is None else value for value in row] for row in dataset["rows"]],
        dtype=float,
    )
    response = matrix[:, columns.index("y")]
    predictors = matrix[:, [columns.index(name) for name in case["predictors"]]]
    return response, predictors


def _options(case, iterations=None):
    if iterations is None:
        iterations = case.get("python_iterations")
    return TwoLevelOptions(
        iterations=iterations,
        intercept=bool(case["intercept"]),
        groupcenter_slope=bool(case["groupcenter_slope"]),
    )


def _fit(case, spec, *, m, iterations=None, seed, observed=None, where=None):
    response, predictors = _response_predictors(spec, case)
    return impute_two_level_normal(
        response,
        predictors,
        np.asarray(case["types"], dtype=float),
        method=case["method"],
        observed=observed,
        where=where,
        options=_options(case, iterations),
        m=m,
        seed=seed,
    )


def _affine_span_matches(actual, expected):
    """Both matrices must have the same shape and span the same column space."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    assert np.linalg.matrix_rank(actual) == actual.shape[1]
    residual = expected - actual @ np.linalg.lstsq(actual, expected, rcond=None)[0]
    np.testing.assert_allclose(residual, 0.0, atol=2e-9)
    residual = actual - expected @ np.linalg.lstsq(expected, actual, rcond=None)[0]
    np.testing.assert_allclose(residual, 0.0, atol=2e-9)


def _python_design(case):
    response, predictors = _response_predictors(STRUCTURE_SPEC, case)
    types = np.asarray(case["types"], dtype=float)
    observed = np.isfinite(response)
    _, groups = np.unique(predictors[:, types == -2][:, 0], return_inverse=True)
    random, fixed = _design(
        predictors, types, groups, observed, int(groups.max()) + 1, case["method"],
        _options(case, iterations=1),
    )
    return observed, random, fixed


def _check_structure(case, reference):
    if reference["r_error"] is not None:
        with pytest.raises(ValueError, match=case["python_error"]):
            _fit(case, STRUCTURE_SPEC, m=1, iterations=1, seed=1)
        return

    observed, random, fixed = _python_design(case)
    if case["method"] == "2l.pan":
        assert fixed is not None
        assert fixed.shape[1] == np.atleast_1d(reference["xcol"]).size
        assert random.shape[1] == np.atleast_1d(reference["zcol"]).size
        _affine_span_matches(fixed[observed], reference["fixed_design"])
        np.testing.assert_allclose(
            random, np.asarray(reference["random_design"], dtype=float), atol=2e-10,
        )
        # The captured pan prior is fixed by mice: a = 1, Binv = 1, c = q,
        # Dinv = I_q.  Its dimensions must agree with the captured design.
        assert np.asarray(reference["prior"]["c"]).item() == random.shape[1]
        assert np.asarray(reference["prior"]["Dinv"]).shape == (
            random.shape[1], random.shape[1],
        )
    else:
        assert fixed is None
        assert random.shape[1] == reference["n_random"]
        np.testing.assert_allclose(
            random[observed], np.asarray(reference["random_design"], dtype=float),
            atol=2e-10,
        )

    response, _ = _response_predictors(STRUCTURE_SPEC, case)
    assert reference["n_imputed"] == len(reference["imputed_rows"])
    assert reference["imputed_rows"] == np.flatnonzero(~np.isfinite(response)).tolist()
    draws = _fit(case, STRUCTURE_SPEC, m=3, iterations=2, seed=23)
    assert draws.shape == (3, reference["n_imputed"])
    assert np.isfinite(draws).all()


def _fit_distribution(case):
    response, predictors = _response_predictors(DISTRIBUTION_SPEC, case)
    observed = np.isfinite(response)
    withheld = case.get("observed_rows")
    if withheld:
        observed = observed.copy()
        observed[np.asarray(withheld, dtype=int)] = False
    targets = case.get("target_rows")
    where = None
    if targets:
        where = np.zeros(len(response), dtype=bool)
        where[np.asarray(targets, dtype=int)] = True
    return _fit(
        case, DISTRIBUTION_SPEC, m=int(case["m"]), seed=int(case["seed"]),
        observed=observed, where=where,
    )


def _check_distribution(case, reference):
    draws = _fit_distribution(case)
    count = draws.shape[0]
    assert draws.shape == (int(case["m"]), len(reference["means"]))
    assert reference["n_draws"] == int(case["m"])

    means = draws.mean(axis=0)
    standard_deviations = draws.std(axis=0, ddof=1)
    centered = draws - means
    covariance = np.cov(draws, rowvar=False, ddof=1)
    products = centered[:, :, None] * centered[:, None, :]
    covariance_se = products.std(axis=0, ddof=1) / np.sqrt(count)

    reference_means = np.asarray(reference["means"], dtype=float)
    reference_sd = np.asarray(reference["standard_deviations"], dtype=float)
    mean_se = np.sqrt((standard_deviations**2 + reference_sd**2) / count)
    np.testing.assert_array_less(
        np.abs(means - reference_means), 5 * mean_se + 1e-12,
        err_msg=f"{case['id']}: row means",
    )
    sd_se = np.sqrt((standard_deviations**2 + reference_sd**2) / (2 * count))
    np.testing.assert_array_less(
        np.abs(standard_deviations - reference_sd), 5 * sd_se + 1e-12,
        err_msg=f"{case['id']}: row standard deviations",
    )
    reference_covariance = np.asarray(reference["covariance"], dtype=float)
    reference_covariance_se = np.asarray(reference["covariance_se"], dtype=float)
    combined_se = np.hypot(covariance_se, reference_covariance_se)
    np.testing.assert_array_less(
        np.abs(covariance - reference_covariance), 5 * combined_se + 1e-12,
        err_msg=f"{case['id']}: target-row covariance",
    )


@pytest.mark.parametrize("case", STRUCTURE_CASES, ids=STRUCTURE_IDS)
def test_deterministic_design_matches_r(case):
    _check_structure(case, STRUCTURE_EXPECTED[case["id"]])


@pytest.mark.parametrize("case", DISTRIBUTION_CASES, ids=DISTRIBUTION_IDS)
def test_stochastic_moments_match_r(case):
    _check_distribution(case, DISTRIBUTION_EXPECTED[case["id"]])


@pytest.mark.parametrize("case_id", ["pan-random-slope", "norm-random-slope"])
def test_two_level_means_in_chained_entry_point(case_id):
    case = next(case for case in DISTRIBUTION_CASES if case["id"] == case_id)
    response, predictors = _response_predictors(DISTRIBUTION_SPEC, case)
    data = np.column_stack((predictors, response, 2 + response))
    outcomes = data.shape[1] - 2
    codes = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    codes[outcomes, :outcomes] = case["types"]
    draws = 350
    result = mice_impute(
        data, methods=[None] * outcomes + [case["method"], lambda x: 2 + x[:, outcomes]],
        predictor_matrix=codes, m=draws, iterations=2, seed=18531,
        two_level_options={outcomes: TwoLevelOptions(iterations=30)},
    )
    actual = result.completed[:, result.where[:, outcomes], outcomes]
    reference = DISTRIBUTION_EXPECTED[case_id]
    mean_se = np.sqrt(actual.var(axis=0, ddof=1) / draws
                      + np.asarray(reference["standard_deviations"])**2 / reference["n_draws"])
    assert (abs(actual.mean(axis=0) - reference["means"]) <= 5 * mean_se).all()
    np.testing.assert_array_equal(result.completed[:, result.where[:, outcomes], -1], 2 + actual)


def test_two_level_chain_uses_current_predictors_and_a_fixed_cluster():
    cluster = np.repeat(np.arange(4), 6)
    x = np.tile([-2.0, -1, 0, 1, 2, 3], 4)
    response = 4 + 2 * x + cluster + np.sin(x)
    response[[2, 5, 11, 17, 23]] = np.nan
    x[2] = np.nan
    data = np.column_stack((cluster, x, response))
    steps = {
        "methods": [None, lambda current: np.where(np.isnan(current[:, 1]), 0, current[:, 1]), "2l.pan"],
        "predictor_matrix": [[0, 0, 0], [0, 0, 0], [-2, 1, 0]],
        "two_level_options": {2: TwoLevelOptions(iterations=10)},
        "m": 2, "iterations": 2, "seed": 321,
    }
    fit = mice_impute(data, **steps)
    assert np.isfinite(fit.completed).all()
    np.testing.assert_array_equal(fit.completed[:, :, 0], np.broadcast_to(cluster, (2, len(data))))
    np.testing.assert_array_equal(fit.completed[:, 2, 1], 0)
    with pytest.raises(ValueError, match="explicit predictor_matrix"):
        mice_impute(data, **{**steps, "predictor_matrix": None})
    with pytest.raises(ValueError, match="one complete cluster"):
        mice_impute(data, **{**steps, "predictor_matrix": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="0/1 indicators"):
        mice_impute(data, **{**steps, "predictor_matrix": [[0, 2, 0], [0, 0, 0], [-2, 1, 0]]})


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
@pytest.mark.parametrize(("experiment", "filename", "cases", "check"), [
    ("structure", "cases.json", STRUCTURE_CASES, _check_structure),
    ("distribution", "distribution_cases.json", DISTRIBUTION_CASES, _check_distribution),
], ids=["structure", "distribution"])
def test_multilevel_live_r(experiment, filename, cases, check):
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), experiment,
         str(FIXTURES / filename)],
        capture_output=True, text=True, check=True, timeout=900,
    )
    references = json.loads(process.stdout)["results"]
    for case in cases:
        check(case, references[case["id"]])


def test_where_and_observed_masks_select_output_rows():
    case = STRUCTURE_CASES[0]
    response, _ = _response_predictors(STRUCTURE_SPEC, case)
    total = len(response)
    observed = np.isfinite(response)
    observed[[1, 2]] = False
    where = np.zeros(total, dtype=bool)
    where[[3, 10, 19]] = True

    draws = _fit(case, STRUCTURE_SPEC, m=4, iterations=5, seed=71,
                 observed=observed, where=where)
    assert draws.shape == (4, 3)
    assert np.isfinite(draws).all()

    # Without ``where`` every row withheld from training becomes a target row.
    draws = _fit(case, STRUCTURE_SPEC, m=2, iterations=5, seed=72, observed=observed)
    assert draws.shape == (2, int((~observed).sum()))
    assert np.isfinite(draws).all()

    # ``where`` may also re-impute observed responses.
    where = np.zeros(total, dtype=bool)
    where[[1, 2]] = True
    draws = _fit(case, STRUCTURE_SPEC, m=3, iterations=5, seed=73, where=where)
    assert draws.shape == (3, 2)
    assert np.isfinite(draws).all()


def test_empty_target_returns_empty_draw_matrix():
    case = STRUCTURE_CASES[0]
    response, predictors = _response_predictors(STRUCTURE_SPEC, case)
    draws = impute_two_level_normal(
        response, predictors, np.asarray(case["types"], dtype=float),
        where=np.zeros(len(response), dtype=bool), m=2, seed=5,
    )
    assert draws.shape == (2, 0)


def test_seed_replays_draws_and_chains_are_independent():
    case = STRUCTURE_CASES[0]
    first = _fit(case, STRUCTURE_SPEC, m=3, iterations=5, seed=101)
    replay = _fit(case, STRUCTURE_SPEC, m=3, iterations=5, seed=101)
    np.testing.assert_array_equal(first, replay)
    other = _fit(case, STRUCTURE_SPEC, m=3, iterations=5, seed=102)
    assert not np.array_equal(first, other)
    assert np.ptp(first, axis=0).min() > 0


def test_default_sampler_options_use_mice_sweep_counts():
    case = STRUCTURE_CASES[0]
    draws = _fit(case, STRUCTURE_SPEC, m=2, seed=11)
    assert draws.shape == (2, 8)
    assert np.isfinite(draws).all()

    norm_case = next(case for case in STRUCTURE_CASES if case["id"] == "norm-random-slope")
    draws = _fit(norm_case, STRUCTURE_SPEC, m=2, seed=11)
    assert draws.shape == (2, 7)
    assert np.isfinite(draws).all()


def _valid_inputs():
    case = STRUCTURE_CASES[0]
    response, predictors = _response_predictors(STRUCTURE_SPEC, case)
    return {
        "y": response.copy(),
        "predictors": predictors.copy(),
        "types": np.asarray(case["types"], dtype=float).copy(),
        "m": 1,
        "seed": 1,
    }


@pytest.mark.parametrize(("overrides", "message"), [
    ({"method": "2l.pmm"}, "method must be"),
    ({"options": {"iterations": 3}}, "options must be"),
    ({"options": TwoLevelOptions(iterations=0)}, "positive integer"),
    ({"options": TwoLevelOptions(iterations=2.5)}, "positive integer"),
    ({"options": TwoLevelOptions(ridge=-1.0)}, "finite and nonnegative"),
    ({"options": TwoLevelOptions(ridge=np.inf)}, "finite and nonnegative"),
    ({"method": "2l.norm", "options": TwoLevelOptions(groupcenter_slope=True)},
     "groupcenter_slope applies"),
    ({"options": TwoLevelOptions(intercept=1.0)}, "must be boolean"),
    ({"m": 0}, "m must be a positive integer"),
    ({"predictors": np.zeros(6)}, "predictors must be"),
    ({"types": np.asarray([-2.0, 2.0])}, "types must code"),
    ({"types": np.asarray([-2.0, 2.0, 5.0])}, "types must code"),
])
def test_sampler_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        impute_two_level_normal(**_valid_inputs() | overrides)


@pytest.mark.parametrize(("modify", "message"), [
    (lambda inputs: inputs["predictors"].__setitem__((0, 1), np.inf), "no infinities"),
    (lambda inputs: inputs["y"].__setitem__(0, np.inf), "no infinities"),
    (lambda inputs: inputs.update(y=inputs["y"][:-1]), "y must be a numeric outcome"),
    (lambda inputs: inputs.update(types=np.asarray([1.0, 1.0, 2.0])),
     "exactly one complete cluster"),
    (lambda inputs: inputs.update(types=np.asarray([-2.0, -2.0, 1.0])),
     "exactly one complete cluster"),
    (lambda inputs: inputs["predictors"].__setitem__((3, 0), np.nan),
     "exactly one complete cluster"),
    (lambda inputs: inputs["predictors"].__setitem__(slice(None), 1.0),
     "at least two clusters"),
])
def test_input_validation(modify, message):
    inputs = _valid_inputs()
    modify(inputs)
    with pytest.raises(ValueError, match=message):
        impute_two_level_normal(**inputs)


def test_incomplete_predictor_at_target_row_is_reported():
    inputs = _valid_inputs()
    missing = np.flatnonzero(~np.isfinite(inputs["y"]))
    assert missing.size
    inputs["predictors"][missing[0], 1] = np.nan
    with pytest.raises(ValueError, match="incomplete predictor"):
        impute_two_level_normal(**inputs)


def test_mask_validation():
    inputs = _valid_inputs()
    with pytest.raises(ValueError, match="observed must identify"):
        impute_two_level_normal(**(inputs | {
            "observed": np.ones(len(inputs["y"]), dtype=int),
        }))
    observed = np.isfinite(inputs["y"])
    observed[np.flatnonzero(~np.isfinite(inputs["y"]))[0]] = True
    with pytest.raises(ValueError, match="observed must identify finite"):
        impute_two_level_normal(**(inputs | {"observed": observed}))
    with pytest.raises(ValueError, match="where must be a boolean"):
        impute_two_level_normal(**(inputs | {
            "where": np.ones(len(inputs["y"]) - 1, dtype=bool),
        }))
    with pytest.raises(ValueError, match="where must be a boolean"):
        impute_two_level_normal(**(inputs | {"where": np.ones(len(inputs["y"]), dtype=int)}))


def test_two_level_pan_needs_a_random_effect():
    inputs = _valid_inputs()
    with pytest.raises(ValueError, match="needs a random intercept"):
        impute_two_level_normal(**(inputs | {
            "types": np.asarray([-2.0, 1.0, 1.0]),
            "options": TwoLevelOptions(intercept=False),
        }))


def test_rank_deficient_fixed_effects_are_reported():
    inputs = _valid_inputs()
    inputs["predictors"][:, 2] = 1.0
    with pytest.raises(ValueError, match="full observed rank"):
        impute_two_level_normal(**(inputs | {
            "types": np.asarray([-2.0, 2.0, 1.0]),
        }))


def test_norm_rejects_fixed_effect_codes_that_mice_silently_ignores():
    inputs = _valid_inputs()
    with pytest.raises(ValueError, match="supports random slopes coded 2"):
        impute_two_level_normal(**(inputs | {
            "method": "2l.norm",
            "types": np.asarray([-2.0, 1.0, 1.0]),
        }))


def test_norm_requires_an_observed_outcome_per_cluster():
    case = next(case for case in STRUCTURE_CASES if case["id"] == "norm-random-slope")
    response, predictors = _response_predictors(STRUCTURE_SPEC, case)
    response[:5] = np.nan
    with pytest.raises(ValueError, match="observed outcome in every cluster"):
        impute_two_level_normal(
            response, predictors, np.asarray(case["types"], dtype=float),
            method="2l.norm", options=TwoLevelOptions(iterations=2), m=1, seed=3,
        )


def test_single_cluster_is_rejected_before_sampling():
    inputs = _valid_inputs()
    inputs["predictors"][:, 0] = 1.0
    with pytest.raises(ValueError, match="at least two clusters"):
        impute_two_level_normal(**inputs)
