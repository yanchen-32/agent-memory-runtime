import csv
import json
from pathlib import Path
import shutil

import pytest

from benchmark import require_frozen_benchmark
from benchmark.freeze_v1 import CHECK_FIELDS, freeze_benchmark
from benchmark.validate_splits import validate_splits


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark" / "data" / "v1.0"


def test_v1_candidate_splits_are_structurally_valid_and_family_disjoint():
    report = validate_splits(DATA_DIR)
    assert report["valid"] is True
    assert report["status"] == "pending_human_review"
    assert report["total_cases"] == 36
    assert report["governance_pairs"] == 6
    assert {item["cases"] for item in report["splits"].values()} == {12}
    manifest = json.loads((DATA_DIR / "candidate_manifest.json").read_text(encoding="utf-8"))
    for split, item in report["splits"].items():
        assert manifest["splits"][split]["sha256"] == item["sha256"]
        assert manifest["splits"][split]["case_count"] == item["cases"]


def test_v1_freeze_refuses_pending_human_review():
    with pytest.raises(ValueError, match="human review incomplete"):
        freeze_benchmark(DATA_DIR, DATA_DIR / "review_checklist.csv")


def test_v1_review_checklist_has_one_row_per_case():
    with (DATA_DIR / "review_checklist.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    assert len({row["case_id"] for row in rows}) == 36


def test_v1_approved_review_freezes_hashes(tmp_path):
    copied = tmp_path / "v1.0"
    shutil.copytree(DATA_DIR, copied)
    review_path = copied / "review_checklist.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = list(rows[0])
    for row in rows:
        for field in CHECK_FIELDS:
            row[field] = "yes"
        row["reviewer"] = "test-reviewer"
        row["decision"] = "approved"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = freeze_benchmark(copied, review_path)
    assert manifest["status"] == "frozen"
    verified = require_frozen_benchmark(copied / "test.jsonl")
    assert verified["reviewed_case_count"] == 36
