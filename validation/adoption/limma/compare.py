"""Compare limma and Python candidates on artificial log abundance data."""
import argparse
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import false_discovery_control


def fixture(unbalanced):
    rng = np.random.default_rng(50462)
    replicates = np.tile([2, 3, 4], 4) if unbalanced else np.full(12, 3)
    block = np.repeat(np.arange(12), replicates)
    group = np.concatenate([np.arange(n) % 2 for n in replicates])
    design = np.column_stack([1 - group, group])
    n_features = 80
    noise_scale = rng.uniform(0.35, 0.9, size=(n_features, 1))
    subject_effect = rng.normal(size=(n_features, 12)) * noise_scale * 1.4
    expression = (
        rng.uniform(5, 11, size=(n_features, 1))
        + rng.normal(0.5, 0.7, size=(n_features, 1)) * group
        + subject_effect[:, block]
        + rng.normal(size=(n_features, block.size)) * noise_scale
    )
    return expression, design, block


def collect(fit, getter):
    fields = ("coefficients", "stdev_unscaled", "sigma", "df_residual",
              "s2_post", "df_total", "t", "p_value")
    values = {("coefficient" if name == "coefficients" else name):
              np.asarray(getter(fit, name), dtype=float).ravel() for name in fields}
    values["adjusted_p"] = false_discovery_control(values["p_value"])
    return values


def inmoose_results(expression, design, block, consensus):
    from inmoose.limma import contrasts_fit, eBayes, lmFit
    from pandas import DataFrame
    from patsy import DesignInfo, DesignMatrix

    design = DesignMatrix(design, DesignInfo(["A", "B"]))
    contrast = DataFrame({"B_vs_A": [-1, 1]}, index=["A", "B"])
    fitted = eBayes(contrasts_fit(lmFit(DataFrame(expression), design=design), contrast))
    try:
        lmFit(DataFrame(expression), design=design, block=block, correlation=consensus)
    except NotImplementedError as error:
        print(f"  blocked fit unavailable: {error}")
    else:
        raise AssertionError("InMoose blocked fitting behavior changed; compare numerics")
    return collect(fitted, getattr)


def pylimma_results(expression, design, block, consensus):
    from pylimma import contrasts_fit, duplicate_correlation, e_bayes, lm_fit

    contrast = np.array([[-1], [1]])
    duplicate = duplicate_correlation(expression, design=design, block=block)
    result = {}
    for label, settings in (
        ("ordinary", {}),
        ("block_fixed", {"block": block, "correlation": consensus}),
        ("block_estimated", {"block": block, "correlation": duplicate["consensus_correlation"]}),
    ):
        fitted = lm_fit(expression, design=design, **settings)
        result[label] = collect(e_bayes(contrasts_fit(fitted, contrasts=contrast)), dict.__getitem__)
    return duplicate, result


def compare(label, output, expected, *, rtol, atol, require_match=False):
    print(f"  {label}")
    for key, actual in output.items():
        reference = expected[key]
        if reference.shape != np.asarray(actual).shape:
            raise ValueError(f"{key} shape mismatch")
        delta = np.abs(actual - reference)
        match = np.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=True)
        print(f"    {key:15s} {np.count_nonzero(match):2d}/{match.size}  max abs {np.nanmax(delta):.9g}")
        if require_match and not match.all():
            raise AssertionError(f"{label} exceeded tolerance in {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=["inmoose", "pylimma", "edgepython"])
    args = parser.parse_args()
    print(f"Python {platform.python_version()}, {args.candidate} {version(args.candidate)}", flush=True)
    oracle = Path(__file__).with_name("reference.R")
    with TemporaryDirectory(prefix="validate-limma-") as workspace:
        folder = Path(workspace)
        for unbalanced in (False, True):
            print("Unbalanced blocks" if unbalanced else "Balanced blocks", flush=True)
            expression, design, block = fixture(unbalanced)
            for name, data in [("expression", expression), ("design", design), ("block", block)]:
                np.savetxt(folder / f"{name}.csv", data, delimiter=",", fmt="%.17g")
            subprocess.run(["Rscript", "--vanilla", str(oracle), workspace], check=True)
            reference = {
                mode: np.genfromtxt(folder / f"{mode}.csv", delimiter=",", names=True)
                for mode in ("ordinary", "block", "duplicates")
            }
            correlation = reference["duplicates"]["consensus"][0]
            if args.candidate == "inmoose":
                results = {"ordinary": inmoose_results(expression, design, block, correlation)}
                duplicate = None
            elif args.candidate == "pylimma":
                duplicate, results = pylimma_results(expression, design, block, correlation)
            else:
                from edgepython import duplicate_correlation
                duplicate = duplicate_correlation(expression, design=design, block=block)
                results = {}

            if duplicate is not None:
                calculated = duplicate["consensus_correlation"]
                print(f"  consensus correlation: Python {calculated:.12g}, R {correlation:.12g}")
                np.testing.assert_allclose(calculated, correlation, rtol=1e-8, atol=1e-10)
                compare("genewise atanh correlation", {"atanh": duplicate["atanh_correlations"]},
                        reference["duplicates"], rtol=1e-8, atol=1e-10, require_match=True)
            for label, result in results.items():
                mode = "ordinary" if label == "ordinary" else "block"
                compare(label, result, reference[mode], rtol=1e-6, atol=1e-9, require_match=True)
        if args.candidate == "pylimma":
            check_quality_weights(folder)


def check_quality_weights(folder):
    from pylimma import array_weights, voom, voom_with_quality_weights

    oracle = Path(__file__).with_name("quality_reference.R")
    rng = np.random.default_rng(25801)
    design = np.column_stack([np.ones(8), np.repeat([0, 1], 4)])
    dispersion_size = np.array([20, 20, 18, 18, 20, 18, 10, 1.8])
    for n_features in (80, 8):
        mu = rng.uniform(10, 70, (n_features, 1)) * np.array([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2])
        counts = rng.negative_binomial(dispersion_size, dispersion_size / (dispersion_size + mu))
        np.savetxt(folder / "counts.csv", counts, delimiter=",", fmt="%d")
        subprocess.run(["Rscript", "--vanilla", str(oracle), str(folder)], check=True)
        fields = ("E", "weights", "sample_weights")
        reference = {field: np.loadtxt(folder / f"{field}.csv", delimiter=",") for field in fields}
        python = voom_with_quality_weights(counts, design)
        actual = {field: python[field] for field in fields}
        compare(f"voomWithQualityWeights ({n_features} features × 8 samples)", actual,
                reference, rtol=1e-6, atol=1e-9, require_match=(n_features != 8))
        if n_features == 8:
            # Spell out limma 3.66.0's square-input weight orientation.
            first = voom(counts, design)
            initial_weights = array_weights(first, design=design)
            second = voom(counts, design, weights=initial_weights[:, None])
            sample_weights = array_weights(second, design=design)
            legacy_output = {
                "E": second["E"], "weights": second["weights"] * sample_weights[None, :],
                "sample_weights": sample_weights,
            }
            compare("explicit limma 3.66.0 square behavior", legacy_output, reference,
                    rtol=1e-6, atol=1e-9, require_match=True)


if __name__ == "__main__":
    main()
