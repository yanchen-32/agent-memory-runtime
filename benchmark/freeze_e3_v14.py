"""Freeze E3 v1.4 Development after one bound human package signoff."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_e3_v14 import SCENARIOS_PER_STRATUM
from .review_e3_v14 import CHECK_FIELDS, REVIEW_VERSION
from .validate_e3_v14 import validate_development


TOOL_FILES = (
    "e3_v14_spec.py",
    "design_e3_v14.py",
    "generate_e3_v14.py",
    "validate_e3_v14.py",
    "review_e3_v14.py",
    "freeze_e3_v14.py",
    "loader.py",
    "metrics.py",
    "statistics.py",
)
METHOD_FILES = (
    "agent/runtime_agent_v2.py",
    "agent/structured_kv.py",
    "agent/clients.py",
    "memory/structured_lookup.py",
    "memory/runtime.py",
    "memory/storage/base.py",
    "memory/storage/in_memory.py",
    "memory/storage/sqlite_store.py",
    "benchmark/runner.py",
    "experiments/run_e3_v14.py",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    development = root / "development.jsonl"
    review_path = root / "review_checklist.csv"
    signoff_path = root / "review_signoff.json"
    audit_path = root / "deep_audit.json"
    audit_md = root / "deep_audit.md"
    candidate_path = root / "candidate_manifest.json"
    validation = validate_development(development)
    with review_path.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    expected_ids = set(validation["case_ids"])
    if {str(row.get("case_id")) for row in reviews} != expected_ids or len(reviews) != len(expected_ids):
        raise ValueError("review checklist must cover every v1.4 case exactly once")
    if not all(row.get(field, "").lower() == "yes" for row in reviews for field in CHECK_FIELDS):
        raise ValueError("technical review is incomplete")
    audit = _json(audit_path)
    if not (
        audit.get("valid")
        and audit.get("status") == "passed_pending_human_review"
        and audit.get("case_count") == len(expected_ids)
        and audit.get("development_sha256") == validation["sha256"]
        and audit.get("review_sha256") == sha256_file(review_path)
    ):
        raise ValueError("deep audit is absent, failed, or stale")
    signoff = _json(signoff_path)
    reviewer = str(signoff.get("reviewer", "")).strip()
    if not reviewer or str(signoff.get("decision", "")).lower() != "approved":
        raise ValueError("fill reviewer once and set decision=approved")
    if not (
        signoff.get("scope") == "all_v14_e3_development_candidates"
        and signoff.get("reviewed_case_count") == len(expected_ids)
        and signoff.get("reviewed_scenario_family_count") == SCENARIOS_PER_STRATUM
        and signoff.get("development_sha256") == validation["sha256"]
        and signoff.get("review_checklist_sha256") == sha256_file(review_path)
        and signoff.get("deep_audit_sha256") == sha256_file(audit_path)
    ):
        raise ValueError("human signoff does not match the audited package")
    candidate = _json(candidate_path)
    if not (
        candidate.get("development_sha256") == validation["sha256"]
        and candidate.get("review_sha256") == sha256_file(review_path)
        and candidate.get("deep_audit", {}).get("sha256") == sha256_file(audit_path)
        and candidate.get("deep_audit", {}).get("report_sha256") == sha256_file(audit_md)
    ):
        raise ValueError("candidate manifest is stale")
    module_root = Path(__file__).resolve().parent
    repository = module_root.parent
    tooling = {name: sha256_file(module_root / name) for name in TOOL_FILES}
    method_candidate = {name: sha256_file(repository / name) for name in METHOD_FILES}
    manifest = {
        "benchmark_version": "1.4-e3",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            "development": {
                "file": development.name,
                "sha256": validation["sha256"],
                "cases": len(expected_ids),
                "case_ids": validation["case_ids"],
            }
        },
        "scenario_family_count": validation["scenario_families"],
        "strata": validation["strata"],
        "token_ranges": validation["token_ranges"],
        "distributions": validation["distributions"],
        "strict_prefix_nested": validation["prefix_nested"],
        "answer_leakage_free": validation["answer_leakage_free"],
        "semantic_surfaces_valid": validation["semantic_surfaces_valid"],
        "review_file": review_path.name,
        "review_sha256": sha256_file(review_path),
        "reviewed_case_count": len(expected_ids),
        "reviewers": [reviewer],
        "review_signoff_file": signoff_path.name,
        "review_signoff_sha256": sha256_file(signoff_path),
        "deep_audit_file": audit_path.name,
        "deep_audit_sha256": sha256_file(audit_path),
        "deep_audit_report_file": audit_md.name,
        "deep_audit_report_sha256": sha256_file(audit_md),
        "deep_audit_version": REVIEW_VERSION,
        "candidate_manifest_file": candidate_path.name,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "tooling_sha256": tooling,
        "method_candidate_sha256": method_candidate,
        "method_status": "candidate_pending_development_admission",
        "test_generated": False,
        "test_accessed": False,
        "holdout_generated": False,
    }
    (root / "frozen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze reviewed v1.4 E3 Development.")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data/v1.4-e3",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
