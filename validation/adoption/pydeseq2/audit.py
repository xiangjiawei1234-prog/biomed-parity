"""Compare PyDESeq2 and DESeq2 on artificial expression count tables."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

DIRECTORY = Path(__file__).resolve().parent
WALD_FIELDS = [
    "baseMean", "dispersion", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj",
]


def artificial_counts(seed=906, n_features=240):
    rng = np.random.default_rng(seed)
    conditions = ["reference"] * 3 + ["tumor"] * 3
    samples = [f"sample_{i + 1}" for i in range(12)]
    metadata = pd.DataFrame(
        {"batch": ["A"] * 6 + ["B"] * 6, "condition": conditions * 2}, index=samples
    )
    ids = [f"feature_{i + 1:03d}" for i in range(n_features)]
    baseline = 2 ** rng.uniform(3, 10, n_features)
    library_factor = rng.uniform(0.78, 1.24, len(samples))
    batch_effect = np.exp(rng.normal(0, 0.18, n_features))
    fold_change = np.ones(n_features)
    fold_change[:30] = rng.uniform(1.35, 2.9, 30)
    fold_change[30:60] = 1 / rng.uniform(1.35, 2.9, 30)
    expected = library_factor[:, None] * baseline[None, :]
    expected[metadata.batch == "B", :] *= batch_effect
    expected[metadata.condition == "tumor", :] *= fold_change
    dispersion = (0.035 + 4 / baseline) * np.exp(rng.normal(0, 0.28, n_features))
    shape = 1 / dispersion
    values = rng.negative_binomial(shape, shape / (shape + expected))
    values[:, -1] = 0
    return pd.DataFrame(values, index=samples, columns=ids), metadata


def dataset(counts, metadata, method="parametric"):
    return DeseqDataSet(
        counts=counts, metadata=metadata, design="~batch + condition", fit_type=method,
        size_factors_fit_type="ratio", refit_cooks=False, n_cpus=1, quiet=True,
    )


def read_r(workspace, name):
    return pd.read_csv(workspace / f"{name}.tsv", sep="\t", index_col=0)


def compare(left, right):
    np.testing.assert_array_equal(left.index, right.index)
    first, second = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    difference = np.abs(first[finite] - second[finite])
    return {
        "matched_finite": int(finite.sum()),
        "finite_mask_disagreements": int((np.isfinite(first) != np.isfinite(second)).sum()),
        "median_absolute_difference": float(np.median(difference)),
        "p95_absolute_difference": float(np.percentile(difference, 95)),
        "max_absolute_difference": float(difference.max()),
    }


def wald_summary(counts, metadata, workspace, fit):
    model = dataset(counts, metadata, method=fit)
    model.deseq2()
    stats = DeseqStats(
        model, contrast=["condition", "tumor", "reference"],
        cooks_filter=False, independent_filter=False, n_cpus=1, quiet=True,
    )
    stats.summary()
    py = stats.results_df.copy()
    py["dispersion"] = model.var["dispersions"]
    r = read_r(workspace, f"{fit}_wald").loc[py.index]
    r_called, py_called = r.padj < 0.05, py.padj < 0.05
    r_factors = read_r(workspace, f"{fit}_sf").loc[metadata.index, "size_factor"]
    return {
        "r_fit_type": (workspace / f"{fit}_type.txt").read_text().strip(),
        "py_fit_type": model.fit_type,
        "size_factor": compare(r_factors, model.obs["size_factors"]),
        "wald": {field: compare(r[field], py[field]) for field in WALD_FIELDS},
        "bh_fdr_0.05": {
            "r_discoveries": int(r_called.sum()),
            "python_discoveries": int(py_called.sum()),
            "disagreement_features": [
                {"id": feature, "r": float(r.padj[feature]), "python": float(py.padj[feature])}
                for feature in py.index[r_called != py_called]
            ],
        },
    }


def compare_vst(counts, metadata, workspace):
    model = dataset(counts, metadata)
    model.vst(use_design=False, fit_type="parametric")
    py = pd.DataFrame(model.layers["vst_counts"], index=counts.index, columns=counts.columns)
    r = read_r(workspace, "parametric_vst").loc[py.index, py.columns]
    return {"parametric": compare(r, py), "python_parametric_fit_type": model.vst_fit_type}


def test_python_feature(operation):
    try:
        operation()
    except (NotImplementedError, TypeError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return "completed"


def main():
    counts, metadata = artificial_counts()
    report = {
        "pydeseq2": version("PyDESeq2"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "pandas": pd.__version__,
        "samples": len(metadata),
        "synthetic_features": counts.shape[1],
        "seed": 906,
        "profiles": {},
    }
    with TemporaryDirectory(prefix="deseq2-audit-") as name:
        workspace = Path(name)
        counts.to_csv(workspace / "counts.tsv", sep="\t")
        metadata.to_csv(workspace / "metadata.tsv", sep="\t")
        report["input_sha256"] = {
            input_name: sha256((workspace / input_name).read_bytes()).hexdigest()
            for input_name in ("counts.tsv", "metadata.tsv")
        }
        r_process = subprocess.run(
            ["Rscript", "--vanilla", str(DIRECTORY / "run_r.R"), str(workspace)],
            capture_output=True, text=True,
        )
        if r_process.returncode:
            raise RuntimeError(r_process.stderr)
        report["r"], report["deseq2"] = (workspace / "r_versions.txt").read_text().splitlines()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for method in ("parametric", "mean"):
                report["profiles"][method] = wald_summary(counts, metadata, workspace, method)
            report["vst"] = compare_vst(counts, metadata, workspace)
            report["vst"]["r_local_shape"] = list(read_r(workspace, "local_vst").shape)
            report["vst"]["python_local"] = test_python_feature(
                lambda: dataset(counts, metadata).vst(use_design=False, fit_type="local")
            )
            report["lrt"] = {
                "r_finite_p_values": int((workspace / "lrt_count.txt").read_text()),
                "python": test_python_feature(
                    lambda: dataset(counts, metadata).deseq2(test="LRT", reduced="~batch")
                ),
            }
        report["python_warnings"] = sorted({str(warning.message) for warning in caught})
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
