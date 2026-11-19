"""Compare the optional pauc distribution with pROC on selected scenarios."""

import csv
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pauc


DIRECTORY = Path(__file__).resolve().parent


def read_tsv(name):
    with (DIRECTORY / name).open() as source:
        return list(csv.DictReader((line for line in source if not line.startswith("#")), delimiter="\t"))


def curve(rows, cohort, key="score_1"):
    cohort_rows = [row for row in rows if int(row["cohort"]) == cohort]
    labels = np.array([int(row["label"]) for row in cohort_rows])
    scores = np.array([float(row[key]) for row in cohort_rows])
    return pauc.ROC(labels, scores)


def main():
    observations = read_tsv("observations.tsv")
    expected_rows = {row["id"]: row for row in read_tsv("expected.tsv")}
    selected = {
        "auc_distinct", "auc_ties", "paired_distinct", "paired_ties",
        "paired_zero_se", "unpaired_unequal", "unpaired_one_constant",
    }
    print(f"# pauc distribution {version('pauc')}")
    output = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    output.writerow(["id", "field", "pROC", "pauc"])
    for scenario in read_tsv("scenarios.tsv"):
        case_id = scenario["id"]
        if case_id not in selected:
            continue
        rows = [row for row in observations if row["dataset"] == scenario["dataset"]]
        first = curve(rows, 1)
        method = scenario["method"]
        if method == "auc":
            lower, upper = pauc.ci_auc(first, method="delong")
            values = {"auc_1": first.auc, "variance_1": pauc.var(first), "lower": lower, "upper": upper}
        else:
            paired = method == "paired"
            second = curve(rows, 1, "score_2") if paired else curve(rows, 2)
            comparison = pauc.compare(first, second, method="delong", paired=paired)
            values = {"statistic": comparison.stat, "p_value": comparison.p_value}
            if paired:
                values["lower"], values["upper"] = comparison.conf_int
        for field, value in values.items():
            output.writerow([case_id, field, expected_rows[case_id][field], f"{value:.17g}"])


if __name__ == "__main__":
    main()
