import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_catalog(filename):
    return json.loads((ROOT / "catalog" / filename).read_text(encoding="utf-8"))


def test_every_researched_method_has_a_status_and_evidence():
    catalog = load_catalog("methods.json")
    progress = load_catalog("progress.json")["methods"]
    methods = {method["id"]: method for method in catalog["methods"]}
    assert len(methods) == len(catalog["methods"])
    assert methods.keys() == progress.keys()
    categories = {category["id"] for category in catalog["categories"]}
    for method_id, method in methods.items():
        assert method["category"] in categories
        assert method["priority"] in {"P0", "P1", "P2"}
        assert method["assessment"] in catalog["assessments"]
        assert method["sources"] and method["validation_targets"]
        item = progress[method_id]
        assert item["covered"] and item["evidence"]
        for path in item["evidence"]:
            assert (ROOT / path).is_file(), path
