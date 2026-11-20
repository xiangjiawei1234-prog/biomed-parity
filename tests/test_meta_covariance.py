import json
import os
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest
from scipy.sparse import issparse

from biomed_parity.meta_covariance import MetaCovarianceWarning, NamedCorrelation, vcalc


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "meta_covariance"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def _inputs(case):
    inputs = {key: value for key, value in case.items() if key != "id"}
    if isinstance(inputs.get("rho"), dict):
        inputs["rho"] = NamedCorrelation(**inputs["rho"])
    return inputs


def _check(case, reference):
    inputs = _inputs(case)
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always", MetaCovarianceWarning)
        values = vcalc(**inputs)
    assert len(observed_warnings) == len(reference["warnings"])
    assert all("not positive definite" in message for message in reference["warnings"])
    np.testing.assert_allclose(values, reference["covariance"], rtol=0, atol=5e-15)
    np.testing.assert_array_equal(np.diag(values), case["vi"])
    np.testing.assert_array_equal(values, values.T)
    compressed = vcalc(**inputs, checkpd=False, sparse=True)
    assert issparse(compressed)
    np.testing.assert_array_equal(compressed.toarray(), values)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_metafor_snapshot(case):
    _check(case, EXPECTED["results"][case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_metafor():
    run = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        capture_output=True, text=True, check=True, timeout=60,
    )
    results = json.loads(run.stdout)["results"]
    for case in CASES:
        _check(case, results[case["id"]])


def test_explicit_blocks_preserve_interleaved_order():
    original = _inputs(next(case for case in CASES if case["id"] == "per-study-correlation-data"))
    permuted = [0, 2, 3, 5, 1, 4]
    sorted_matrix = vcalc(**original)
    actual = vcalc(**{name: np.asarray(value)[permuted] for name, value in original.items()})
    np.testing.assert_array_equal(actual, sorted_matrix[np.ix_(permuted, permuted)])


@pytest.mark.parametrize("options, match", [
    ({"obs": ["a", "b"]}, "rho is required"),
    ({"type": ["a", "b"], "obs": ["a", "b"], "rho": 0.3}, "both within- and between-construct"),
    ({"grp2": ["a", "b"]}, "grp2 requires grp1"),
    ({"grp1": ["a", "a"], "grp2": ["a", "a"]}, "zero model variance"),
    ({"time1": [0, 0.4], "phi": -0.2}, "integral time differences"),
    ({"w1": [1, 0]}, "arm weights must be positive"),
    ({"rvars": [[1, np.nan], [0.1, 1]], "phi": 0.5}, "rvars cannot be combined"),
])
def test_invalid_meta_sampling_covariance(options, match):
    with pytest.raises(ValueError, match=match):
        vcalc([0.1, 0.2], [1, 1], **options)
