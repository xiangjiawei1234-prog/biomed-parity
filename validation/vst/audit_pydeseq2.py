"""Test the optional upstream PyDESeq2 local VST on a shared fixture."""

from importlib.metadata import distribution
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2 import __version__
from pydeseq2.dds import DeseqDataSet

from biomed_parity.vst import local_vst


FIXTURES = Path(__file__).resolve().parent


def main():
    fixture = json.loads((FIXTURES / "cases.json").read_text())
    expected = json.loads((FIXTURES / "expected.json").read_text())["results"]["fitted-locfit"]
    counts = np.asarray(fixture["counts"]["fitted"])
    samples = [f"S{i}" for i in range(counts.shape[1])]
    features = [f"F{i}" for i in range(counts.shape[0])]
    dds = DeseqDataSet(
        counts=pd.DataFrame(counts.T, index=samples, columns=features),
        metadata=pd.DataFrame(index=samples), design="~1", n_cpus=1, quiet=True,
    )
    installed_url = distribution("pydeseq2").read_text("direct_url.json")
    report = {
        "pydeseq2_version": __version__,
        "commit": json.loads(installed_url).get("vcs_info", {}).get("commit_id") if installed_url else None,
        "shape_feature_by_sample": list(counts.shape),
    }
    try:
        dds.vst(fit_type="local")
    except NotImplementedError as error:
        report["failure"] = str(error)
    else:
        from pydeseq2.stats import predict_local_regression

        transform = dds.layers["vst_counts"].T
        size_factors = np.asarray(dds.obs["size_factors"])

        def fitted_candidate_trend(means):
            return np.exp(predict_local_regression(dds.uns["vst_local_trend"], np.log(means)))

        with_candidate_trend = local_vst(counts, fitted_candidate_trend, size_factors=size_factors)
        reference_difference = np.abs(transform - expected["transform"])
        measurements = {
            "size_factor_max_abs_difference_vs_r": float(
                np.max(np.abs(size_factors - expected["size_factors"]))
            ),
            "transform_max_abs_difference_vs_r": float(reference_difference.max()),
            "transform_median_abs_difference_vs_r": float(np.median(reference_difference)),
            "transform_max_abs_difference_with_same_python_fitted_curve": float(
                np.max(np.abs(transform - with_candidate_trend))
            ),
        }
        if not np.isfinite(list(measurements.values())).all():
            raise ValueError("candidate produced nonfinite results")
        report.update(measurements)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
