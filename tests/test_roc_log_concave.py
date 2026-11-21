import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from biomed_parity.roc_smoothing import smooth_roc


FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "roc_log_concave"
INPUT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
CASES = INPUT["cases"]
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["results"]


def _assert_parity(case, reference):
    data = INPUT["datasets"][case["dataset"]]
    options = {key: val for key, val in case.items() if key not in ("id", "dataset", "error")}
    if case.get("error"):
        assert "error" in reference
        with pytest.raises(ValueError, match="two distinct scores in each class"):
            smooth_roc(**data, **options)
        return

    fit = smooth_roc(**data, **options)
    assert fit.coefficients is fit.bandwidth is None
    assert fit.direction == reference["direction"]
    assert fit.method == case["method"]
    assert fit.percent == options.get("percent", False)
    assert not reference["warnings"]
    assert fit.specificity.size == fit.sensitivity.size == case["n"] + 2
    scale = 100 if fit.percent else 1
    np.testing.assert_allclose(fit.specificity, reference["specificity"], rtol=0, atol=2e-15 * scale)
    np.testing.assert_allclose(fit.sensitivity, reference["sensitivity"], rtol=0, atol=3e-11 * scale)
    assert fit.auc == pytest.approx(reference["auc"], rel=0, abs=3e-11 * scale)
    np.testing.assert_array_equal(fit.specificity[[0, 1, -2, -1]], [0, 0, scale, scale])
    np.testing.assert_array_equal(fit.sensitivity[[0, 1, -2, -1]], [scale, scale, 0, 0])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_matches_recorded_p_roc(case):
    _assert_parity(case, EXPECTED[case["id"]])


@pytest.mark.skipif(os.environ.get("BIOMED_R_PARITY") != "1", reason="live R parity is opt-in")
def test_matches_live_p_roc():
    process = subprocess.run(
        ["Rscript", "--vanilla", str(FIXTURES / "oracle.R"), str(FIXTURES / "cases.json")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    references = json.loads(process.stdout)["results"]
    for case in CASES:
        _assert_parity(case, references[case["id"]])


@pytest.mark.parametrize("method", ["logcondens", "logcondens.smooth"])
def test_p_roc_curve_uses_high_scores_even_in_reverse_direction(method):
    data = INPUT["datasets"]["different_supports"]
    forward = smooth_roc(**data, method=method, direction="<", n=17)
    backward = smooth_roc(**data, method=method, direction=">", n=17)
    assert backward.direction == ">"
    np.testing.assert_array_equal(forward.sensitivity, backward.sensitivity)
    assert forward.auc == backward.auc
    with pytest.raises(ValueError, match="at least 2"):
        smooth_roc(**data, method=method, n=1)
