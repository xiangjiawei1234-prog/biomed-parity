"""Generate reproducible synthetic repeated-measure event counts."""

import csv
from pathlib import Path

import numpy as np
from scipy.special import expit


def main():
    options = [
        ("intercept", 8542, 24, 16, (0.85, 0), (-0.75, 0), 0.60, False),
        ("covariates", 6062, 36, 18, (0.72, 0.55), (-1.05, 0.80), 0.68, True),
        ("sparse_groups", 5672, 30, 8, (0.81, 0.40), (-0.30, 0.51), 0.72, True),
    ]
    with Path(__file__).with_name("observations.tsv").open("w", newline="") as target:
        output = csv.writer(target, delimiter="\t", lineterminator="\n")
        output.writerow(["dataset", "group", "count", "x", "z", "offset"])
        for dataset, seed, n_groups, n_per_group, beta, gamma, sd, covariates in options:
            rng = np.random.default_rng(seed)
            group = np.repeat(np.arange(n_groups), n_per_group)
            x = rng.integers(0, 2, group.size) if covariates else np.zeros(group.size, dtype=int)
            z = rng.integers(0, 2, group.size) if covariates else np.zeros(group.size, dtype=int)
            offset = rng.choice([-0.25, 0.0, 0.25], group.size) if covariates else np.zeros(group.size)
            group_effect = rng.normal(0, sd, n_groups)
            eta = beta[0] + beta[1] * x + group_effect[group] + offset
            pi = expit(gamma[0] + gamma[1] * z)
            counts = rng.poisson(np.exp(eta))
            counts[rng.uniform(size=counts.size) < pi] = 0
            output.writerows(zip(
                [dataset] * group.size, group, counts, x, z, offset, strict=True,
            ))


if __name__ == "__main__":
    main()
