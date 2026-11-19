"""Compare edgeR quasi-likelihood fitting against an optional Python candidate."""
import argparse
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def fixtures():
    rng = np.random.default_rng(21260)
    sizes = rng.uniform(2.3, 26, size=(96, 1))
    mu = rng.uniform(25, 120, size=(96, 1)) * np.array([1, 1, 1, 1, 1.15, 1.15, 1.15, 1.15])
    ordinary = rng.negative_binomial(sizes, sizes / (sizes + mu))

    mu_low = rng.uniform(0.9, 7, size=(96, 1)) * np.array([1, 1, 1, 1, 1.3, 1.3, 1.3, 1.3])
    small = rng.negative_binomial(sizes, sizes / (sizes + mu_low)).astype(float)
    small *= rng.choice([0.25, 0.5, 1.0], size=small.shape)
    small[0] = [0, 0.25, 0, 0.5, 0.25, 0, 0.75, 0]
    return {"ordinary_integer": ordinary, "small_fractional": small}


def inmoose_result(counts, design, library_size):
    from inmoose import edgepy
    from pandas import DataFrame
    from patsy import DesignInfo, DesignMatrix

    design = DesignMatrix(design, DesignInfo(["Intercept", "Group"]))
    fitted = edgepy.glmQLFit(DataFrame(counts), design=design, dispersion=0.14, lib_size=library_size)
    report = edgepy.glmQLFTest(fitted, coef=1)
    return dict(
        logFC=report["log2FoldChange"], logCPM=report["logCPM"], F=report["stat"],
        PValue=report["pvalue"], df_effective=fitted.df_residual_zeros, var_post=fitted.var_post,
        dispersion=np.broadcast_to(np.asarray(fitted.dispersion).ravel(), (len(counts),)),
    )


def edgepython_result(counts, design, library_size, *, mode):
    from edgepython import glm_ql_fit, glm_ql_ftest

    legacy = mode == "legacy"
    dispersion = None if mode == "v4_auto" else 0.14
    fitted = glm_ql_fit(counts, design=design, dispersion=dispersion, lib_size=library_size, legacy=legacy)
    tested = glm_ql_ftest(fitted, coef=1)
    table = tested["table"]
    df_name = "df.residual.zeros" if legacy else "df.residual.adj"
    return {**{name: table[name] for name in ("logFC", "logCPM", "F", "PValue")},
            "df_effective": fitted[df_name], "var_post": fitted["s2.post"],
            "dispersion": np.broadcast_to(np.asarray(fitted["dispersion"]).ravel(), (len(counts),))}


def report_differences(label, actual, expected, *, require_match=False):
    if set(actual) != set(expected.dtype.names):
        raise ValueError("Output column mismatch")
    print(f"  {label}")
    for field, measured in actual.items():
        predicted = np.asarray(measured, dtype=float)
        wanted = expected[field]
        if predicted.shape != wanted.shape:
            raise ValueError(f"Incompatible shapes for {field}: {predicted.shape}, {wanted.shape}")
        finite = np.isfinite(predicted) & np.isfinite(wanted)
        maximum = np.max(np.abs(predicted[finite] - wanted[finite])) if finite.any() else np.nan
        agrees = np.isclose(predicted, wanted, rtol=1e-6, atol=1e-9, equal_nan=True)
        print(f"    {field:14s} matched {np.count_nonzero(agrees):3d}/{agrees.size:3d}  max abs {maximum:.9g}")
        if require_match and not agrees.all():
            raise AssertionError(f"{label} exceeded tolerance in {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("inmoose", "edgepython"))
    args = parser.parse_args()
    print(f"Python {platform.python_version()}, {args.candidate} {version(args.candidate)}", flush=True)
    design = np.column_stack([np.ones(8), np.repeat([0, 1], 4)])
    library_size = np.full(8, 1000.0)
    script = Path(__file__).with_name("reference.R")
    with TemporaryDirectory(prefix="validate-edger-") as work:
        path = Path(work)
        for label, counts in fixtures().items():
            print(f"{label} ({counts.shape[0]} features × {counts.shape[1]} samples)", flush=True)
            np.savetxt(path / f"{label}.csv", counts, delimiter=",", fmt="%.17g")
            subprocess.run(["Rscript", "--vanilla", str(script), work, label], check=True)
            expected = {
                mode: np.genfromtxt(path / f"{label}.{mode}.csv", delimiter=",", names=True)
                for mode in ("legacy", "v4", "v4_auto")
            }
            if args.candidate == "inmoose":
                actual = inmoose_result(counts, design, library_size)
                for mode in ("legacy", "v4"):
                    report_differences(f"InMoose vs edgeR {mode}", actual, expected[mode],
                                       require_match=(mode == "legacy"))
            else:
                for mode, r_result in expected.items():
                    actual = edgepython_result(counts, design, library_size, mode=mode)
                    report_differences(f"edgePython vs edgeR {mode}", actual, r_result,
                                       require_match=True)


if __name__ == "__main__":
    main()
