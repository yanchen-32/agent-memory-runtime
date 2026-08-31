import csv
import json
from pathlib import Path
import shutil

import pytest

from benchmark.freeze_v1 import CHECK_FIELDS
from benchmark.generate_v11 import (
    DEFAULT_SEED,
    build_all_splits,
    build_development,
    write_all_splits,
    write_development,
)
from benchmark.audit_v11 import write_audit
from benchmark.freeze_v11 import freeze_benchmark_v11
from benchmark.prereview_v11 import prereview_development
from benchmark.validate_v11 import (
    DEVELOPMENT_CATEGORIES,
    validate_candidate_splits,
    validate_development,
)
from agent import RuleBasedClient
from benchmark import load_jsonl
from benchmark.runner import run_case
from memory import HashEmbeddingModel
from benchmark import require_frozen_benchmark


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "benchmark" / "data" / "v1.1" / "source_manifest.json"


def _approve_signoff(signoff_path: Path) -> None:
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    signoff["reviewer"] = "test-reviewer"
    signoff["decision"] = "approved"
    signoff_path.write_text(
        json.dumps(signoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_v11_development_generator_is_deterministic():
    first = build_development(DEFAULT_SEED)
    second = build_development(DEFAULT_SEED)
    assert first == second
    assert len(first) == 288
    assert sum(case["category"] == "temporal" for case in first) == 48
    assert sum("pair_id" in case["metadata"] for case in first) == 96


def test_v11_generated_development_validates_and_has_blank_human_review(tmp_path):
    manifest = write_development(tmp_path, DEFAULT_SEED)
    report = validate_development(tmp_path / "development.jsonl")
    assert report["status"] == "pending_human_review"
    assert report["governance_pairs"] == 48
    assert report["categories"] == dict(sorted(DEVELOPMENT_CATEGORIES.items()))
    assert report["sha256"] == manifest["development_sha256"]
    with (tmp_path / "review_checklist.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 288
    assert all("reviewer" not in row and "decision" not in row for row in rows)
    assert all(row["notes"] == "pending-technical-prereview" for row in rows)


def test_v11_pairs_have_explicit_closed_old_interval():
    pair = build_development(DEFAULT_SEED)[:2]
    current, historical = pair
    old, new = current["conversation"]
    assert old["valid_to"] == new["valid_from"]
    assert "valid_to" not in new
    assert current["forbidden_memory_ids"] == [old["memory_id"]]
    assert historical["forbidden_memory_ids"] == [new["memory_id"]]


def test_v11_long_context_cases_have_realistic_memory_depth():
    long_cases = [
        case for case in build_development(DEFAULT_SEED) if case["category"] == "long_context"
    ]
    assert len(long_cases) == 24
    assert all(len(case["conversation"]) == 64 for case in long_cases)
    assert all(sum(len(turn["content"]) for turn in case["conversation"]) >= 1_500 for case in long_cases)


def test_v11_prereview_records_technical_checks_without_human_signature(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    report = prereview_development(tmp_path)
    assert report["technical_failures"] == 0
    with (tmp_path / "review_checklist.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 480
    assert all(row[field] == "yes" for row in rows for field in CHECK_FIELDS)
    assert all("reviewer" not in row and "decision" not in row for row in rows)
    signoff = json.loads((tmp_path / "review_signoff.json").read_text(encoding="utf-8"))
    assert signoff["reviewer"] == ""
    assert signoff["decision"] == "pending_human_review"
    assert signoff["reviewed_case_count"] == 480


def test_v11_full_candidate_has_disjoint_splits_and_reasonable_size(tmp_path):
    manifest = write_all_splits(tmp_path, DEFAULT_SEED)
    report = validate_candidate_splits(tmp_path)
    assert manifest["total_case_count"] == 480
    assert report["total_cases"] == 480
    assert report["governance_pairs"] == 80
    assert {item["cases"] for item in report["splits"].values()} == {96, 288}
    all_splits = build_all_splits(DEFAULT_SEED)
    development_families = {case["metadata"]["scenario_family"] for case in all_splits["development"]}
    test_families = {case["metadata"]["scenario_family"] for case in all_splits["test"]}
    holdout_families = {case["metadata"]["scenario_family"] for case in all_splits["holdout"]}
    assert not (development_families & test_families)
    assert not (development_families & holdout_families)
    assert not (test_families & holdout_families)


def test_v11_all_budget_cases_keep_every_comparable_agent_within_80_tokens(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    budgets = [
        case
        for split in ("development", "test", "holdout")
        for case in load_jsonl(tmp_path / f"{split}.jsonl")
        if case.category == "budget"
    ]
    assert len(budgets) == 40
    for agent_name in ("B1", "B2", "B3", "Ours"):
        rows = [
            run_case(agent_name, case, RuleBasedClient, lambda: HashEmbeddingModel(dim=64))
            for case in budgets
        ]
        assert all(row["budget_satisfied"] for row in rows)
        assert max(row["budget_after_prompt_tokens"] for row in rows) <= 80


def test_v11_deep_audit_passes_after_generation_and_prereview(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    prereview_development(tmp_path)
    report = write_audit(tmp_path)
    assert report["valid"] is True
    assert report["case_count"] == 480
    assert report["checks"]["max_normalized_shell_group"] == 4
    assert report["checks"]["long_context_memory_depth_min"] == 64
    assert report["checks"]["ambiguous_non_evidence_answers"] == 0


def test_v11_freeze_refuses_unsigned_review(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    prereview_development(tmp_path)
    write_audit(tmp_path)
    with pytest.raises(ValueError, match="one-time human review signoff incomplete"):
        freeze_benchmark_v11(tmp_path)


def test_v11_complete_human_review_freezes_and_authorizes_formal_split(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    prereview_development(tmp_path)
    write_audit(tmp_path)
    _approve_signoff(tmp_path / "review_signoff.json")
    shutil.copy(SOURCE_MANIFEST, tmp_path / "source_manifest.json")
    manifest = freeze_benchmark_v11(tmp_path)
    assert manifest["status"] == "frozen"
    assert manifest["reviewed_case_count"] == 480
    assert manifest["reviewers"] == ["test-reviewer"]
    verified = require_frozen_benchmark(tmp_path / "test.jsonl")
    assert verified["benchmark_version"] == "1.1"

    signoff_path = tmp_path / "review_signoff.json"
    signoff_path.write_text(signoff_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="human review signoff no longer matches"):
        require_frozen_benchmark(tmp_path / "test.jsonl")


def test_v11_freeze_rejects_split_changed_after_deep_audit(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    prereview_development(tmp_path)
    write_audit(tmp_path)
    _approve_signoff(tmp_path / "review_signoff.json")
    shutil.copy(SOURCE_MANIFEST, tmp_path / "source_manifest.json")
    development = tmp_path / "development.jsonl"
    development.write_text(development.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split changed after deep audit"):
        freeze_benchmark_v11(tmp_path)


def test_v11_freeze_rejects_checklist_changed_after_one_time_signoff(tmp_path):
    write_all_splits(tmp_path, DEFAULT_SEED)
    prereview_development(tmp_path)
    write_audit(tmp_path)
    _approve_signoff(tmp_path / "review_signoff.json")
    shutil.copy(SOURCE_MANIFEST, tmp_path / "source_manifest.json")
    review_path = tmp_path / "review_checklist.csv"
    review_path.write_text(review_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="review checklist changed after one-time human signoff"):
        freeze_benchmark_v11(tmp_path)
