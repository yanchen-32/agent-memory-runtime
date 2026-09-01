import csv
import json
from pathlib import Path

import pytest

from benchmark import require_frozen_benchmark
from benchmark.audit_v12 import write_audit
from benchmark.freeze_v12 import freeze_benchmark_v12
from benchmark.generate_v12 import build_all_splits, write_all_splits
from benchmark.prereview_v12 import CHECK_FIELDS, prereview_candidates
from benchmark.validate_v12 import validate_candidate_splits


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "benchmark/data/v1.1"
PROTOCOL = ROOT / "docs/answer_scoring_protocol_v1.2.md"


def _generate(target: Path) -> None:
    write_all_splits(target, PARENT, scoring_protocol=PROTOCOL)


def _approve(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reviewer"] = "test-reviewer"
    payload["decision"] = "approved"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_v12_is_a_deterministic_scoring_only_migration():
    first = build_all_splits(PARENT)
    second = build_all_splits(PARENT)
    assert first == second
    assert sum(len(cases) for cases in first.values()) == 480
    assert sum(
        case["category"] == "budget" and "answer_spec" in case
        for cases in first.values()
        for case in cases
    ) == 40
    assert all(
        "answer_spec" not in case
        for cases in first.values()
        for case in cases
        if case["category"] != "budget"
    )


def test_v12_candidate_validates_and_prereviews_all_specs(tmp_path):
    _generate(tmp_path)
    validation = validate_candidate_splits(tmp_path)
    prereview = prereview_candidates(tmp_path)
    assert validation["total_cases"] == 480
    assert validation["budget_answer_specs"] == 40
    assert prereview["checked_budget_answer_specs"] == 40
    with (tmp_path / "review_checklist.csv").open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    assert len(reviews) == 480
    assert all(row[field] == "yes" for row in reviews for field in CHECK_FIELDS)


def test_v12_audit_proves_no_undeclared_parent_changes(tmp_path):
    _generate(tmp_path)
    prereview_candidates(tmp_path)
    report = write_audit(tmp_path, PARENT)
    assert report["valid"] is True
    assert report["parent_equivalent_case_count"] == 480
    assert report["unauthorized_change_count"] == 0


def test_v12_freeze_requires_one_human_signoff_and_authorizes_development(tmp_path):
    _generate(tmp_path)
    prereview_candidates(tmp_path)
    write_audit(tmp_path, PARENT)
    with pytest.raises(ValueError, match="one-time human review signoff incomplete"):
        freeze_benchmark_v12(tmp_path, PARENT)
    _approve(tmp_path / "review_signoff.json")
    manifest = freeze_benchmark_v12(tmp_path, PARENT)
    assert manifest["benchmark_version"] == "1.2"
    assert manifest["answer_scorer_version"] == "quantity-semantic-v1"
    assert manifest["reviewers"] == ["test-reviewer"]
    assert require_frozen_benchmark(tmp_path / "development.jsonl")["status"] == "frozen"


def test_v12_audit_rejects_any_change_outside_declared_migration(tmp_path):
    _generate(tmp_path)
    prereview_candidates(tmp_path)
    development = tmp_path / "development.jsonl"
    cases = [json.loads(line) for line in development.read_text(encoding="utf-8").splitlines()]
    cases[0]["query"] = "被篡改的问题？"
    development.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    report = write_audit(tmp_path, PARENT)
    assert report["valid"] is False
    assert report["unauthorized_change_count"] == 1
