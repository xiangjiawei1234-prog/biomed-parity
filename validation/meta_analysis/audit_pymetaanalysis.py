"""Compare optional PyMetaAnalysis with selected metafor oracle outputs."""

from collections import defaultdict
import csv
from importlib.metadata import version
from pathlib import Path
import sys

from meta_analyze import meta_analysis


DIRECTORY = Path(__file__).resolve().parent


def read_tsv(name):
    with (DIRECTORY / name).open() as source:
        return list(csv.DictReader((line for line in source if not line.startswith("#")), delimiter="\t"))


def main():
    data = defaultdict(list)
    for row in read_tsv("observations.tsv"):
        data[row["dataset"]].append((float(row["effect"]), float(row["variance"])))
    expected = {row["id"]: row for row in read_tsv("expected.tsv")}
    print(f"# PyMetaAnalysis distribution {version('PyMetaAnalysis')}")
    output = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    output.writerow(["id", "field", "metafor", "PyMetaAnalysis", "reference"])
    scenarios = [
        ("heterogeneous", "heterogeneous_knha", "heterogeneous_knha_riley"),
        ("boundary", "boundary_knha", "boundary_knha_riley"),
        ("two_small", "two_small", "two_small"),
    ]
    for dataset, primary_id, riley_id in scenarios:
        effects, variances = zip(*data[dataset], strict=True)
        observed = meta_analysis(effect=effects, variance=variances, ci_method="hartung_knapp")
        profile = observed.tau2_confidence_interval()
        fields = [
            ("mean_effect", observed.estimate, primary_id, "REML"),
            ("tau2", observed.tau2, primary_id, "REML"),
            ("lower", observed.ci_low, primary_id, "Knapp-Hartung"),
            ("upper", observed.ci_high, primary_id, "Knapp-Hartung"),
            ("tau2_lower", profile.ci_low, primary_id, f"PL vs {profile.method}"),
            ("tau2_upper", profile.ci_high, primary_id, f"PL vs {profile.method}"),
        ]
        if observed.prediction_interval is not None:
            lower, upper = observed.prediction_interval
            fields.extend([
                ("prediction_lower", lower, riley_id, "Riley"),
                ("prediction_upper", upper, riley_id, "Riley"),
            ])
        else:
            fields.extend([
                ("prediction_lower", None, riley_id, "Riley"),
                ("prediction_upper", None, riley_id, "Riley"),
            ])
        for field, value, reference_id, method in fields:
            value_text = "NA" if value is None else f"{value:.17g}"
            output.writerow([reference_id, field, expected[reference_id][field], value_text, method])


if __name__ == "__main__":
    main()
