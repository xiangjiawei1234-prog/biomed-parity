"""Compare the installed Python package against the local R survival package."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from biomed_parity.survival import survdiff, survdiff_one_sample

HERE = Path(__file__).resolve().parent


def assert_case(case, expected):
    arguments = {key: value for key, value in case.items() if key != "id"}
    if "error" in expected:
        try:
            survdiff(**arguments)
        except np.linalg.LinAlgError:
            return
        raise AssertionError("expected a singular covariance error")
    fit = (
        survdiff_one_sample(**arguments)
        if "survival_probability" in arguments
        else survdiff(**arguments)
    )
    for name, value in expected.items():
        if name == "id":
            continue
        actual = getattr(fit, name)
        if value is None:
            assert actual is None, name
        elif name in {"groups", "strata"}:
            assert actual == tuple(value), name
        elif name in {"counts", "strata_counts", "n", "df"}:
            np.testing.assert_array_equal(actual, value, err_msg=name)
        elif name == "pvalue":
            np.testing.assert_allclose(actual, float(value), rtol=1e-10, atol=0, err_msg=name)
        else:
            if isinstance(value, str):
                value = float(value)
            np.testing.assert_allclose(actual, value, rtol=1e-11, atol=5e-13, err_msg=name)


def random_cases(count):
    generator = np.random.default_rng(20261119)
    for index in range(count):
        size = int(generator.integers(20, 90))
        groups = int(generator.integers(2, 6))
        group = generator.integers(0, groups, size)
        group[:groups] = np.arange(groups)
        strata = generator.integers(0, int(generator.integers(2, 6)), size)
        time = generator.integers(0, 20, size).astype(float)
        time += generator.integers(0, 3, size) * 1e-9
        yield {
            "id": f"generated_{index:03d}",
            "time": time.tolist(),
            "event": generator.integers(0, 2, size).tolist(),
            "group": group.tolist(),
            "strata": strata.tolist() if index % 2 else None,
            "rho": [-1, 0, 0.5, 1, 2][index % 5],
            "timefix": index % 3 != 0,
        }


def verify(rscript="Rscript", random_count=0):
    inputs = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    inputs.extend(random_cases(random_count))
    with TemporaryDirectory() as folder:
        path = Path(folder) / "cases.json"
        path.write_text(json.dumps(inputs), encoding="utf-8")
        reference = subprocess.run(
            [rscript, "--vanilla", str(HERE / "oracle.R"), str(path)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    reference = json.loads(reference.stdout)
    expected = {case["id"]: case for case in reference["cases"]}
    assert set(expected) == {case["id"] for case in inputs}
    for case in inputs:
        try:
            assert_case(case, expected[case["id"]])
        except AssertionError as error:
            raise AssertionError(f"survdiff parity failed for {case['id']}: {error}") from error
    return {"method": "survival::survdiff", "cases_passed": len(inputs), **reference["versions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--random-cases", default=0, type=int)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be nonnegative")
    print(json.dumps(verify(args.rscript, args.random_cases)))


if __name__ == "__main__":
    main()
