import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from biomed_parity.nbinom_lrt import nbinom_lrt

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "nbinom_lrt"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _check_case(case, expected):
    actual = asdict(nbinom_lrt(**{key: value for key, value in case.items() if key != "id"}))
    assert actual.keys() == expected.keys()
    for field, value in actual.items():
        reference = expected[field]
        if isinstance(reference, list):
            reference = [np.nan if x == "NA" else x for x in reference]
        np.testing.assert_allclose(
            value, reference, rtol=5e-8, atol=8e-8, err_msg=f"{case['id']}.{field}",
        )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_matches_recorded_r(case):
    _check_case(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_r():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    results = json.loads(process.stdout)["results"]
    for case in CASES:
        _check_case(case, results[case["id"]])


def test_column_basis_and_sample_order_invariance():
    base = {key: value for key, value in CASES[0].items() if key != "id"}
    rng = np.random.default_rng(11)
    subjects = rng.permutation(len(base["size_factors"]))
    updated = dict(base)
    updated["counts"] = np.asarray(base["counts"])[:, subjects]
    updated["size_factors"] = np.asarray(base["size_factors"])[subjects]
    updated["full"] = np.asarray(base["full"])[subjects][:, ::-1]
    updated["reduced"] = np.asarray(base["reduced"])[subjects][:, ::-1]
    comparison = nbinom_lrt(**updated)
    reference = nbinom_lrt(**base)
    np.testing.assert_allclose(comparison.statistic, reference.statistic, atol=1e-9)


def test_missing_dispersion_is_allowed_for_all_zero_feature():
    full = [[1, 0], [1, 0], [1, 1], [1, 1]]
    result = nbinom_lrt([[0, 0, 0, 0]], full, [[1]] * 4, [np.nan], [1] * 4)
    assert not result.tested[0]
    assert np.isnan(result.adjusted_pvalue[0])


@pytest.mark.parametrize(
    "options,error",
    [
        ({"counts": [[1, 2.5, 3, 4]]}, "nonnegative integers"),
        ({"reduced": [[0], [1], [0], [1]]}, "strictly nested"),
        ({"dispersions": [-0.2]}, "positive value"),
        ({"size_factors": [1, 1, 1, 0]}, "positive"),
        ({"full": [[1, 1]] * 4}, "full column rank"),
    ],
)
def test_invalid_inputs(options, error):
    input_data = {
        "counts": [[1, 2, 3, 4]],
        "full": [[1, 0], [1, 0], [1, 1], [1, 1]],
        "reduced": [[1]] * 4,
        "dispersions": [0.1],
        "size_factors": [1] * 4,
    }
    with pytest.raises(ValueError, match=error):
        nbinom_lrt(**(input_data | options))
