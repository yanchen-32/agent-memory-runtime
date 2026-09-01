"""Deep migration audit for Benchmark v1.2 scoring candidates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

from .artifacts import sha256_file
from .execution import require_frozen_benchmark
from .generate_v12 import REVIEW_FIELDS, _read_jsonl, upgrade_case
from .prereview_v12 import CHECK_FIELDS, write_signoff_template
from .validate_v12 import SPLITS, validate_candidate_splits


AUDIT_VERSION = "v1.2.0"


def audit_candidates(data_dir: str | Path, parent_dir: str | Path) -> dict:
    root = Path(data_dir)
    parent_root = Path(parent_dir)
    validation = validate_candidate_splits(root)
    failures = []
    changed_case_count = 0
    unauthorized_changes = []
    parent_hashes = {}
    for split in SPLITS:
        parent_path = parent_root / f"{split}.jsonl"
        require_frozen_benchmark(parent_path)
        parent_hashes[split] = sha256_file(parent_path)
        parents = _read_jsonl(parent_path)
        candidates = _read_jsonl(root / f"{split}.jsonl")
        if len(parents) != len(candidates):
            failures.append(f"{split}: parent/candidate count mismatch")
            continue
        for parent, candidate in zip(parents, candidates, strict=True):
            if parent.get("case_id") != candidate.get("case_id"):
                unauthorized_changes.append(str(candidate.get("case_id")))
            elif candidate != upgrade_case(parent):
                unauthorized_changes.append(str(candidate["case_id"]))
            else:
                changed_case_count += 1
    if unauthorized_changes:
        failures.append(f"unauthorized case changes: {unauthorized_changes[:10]}")

    review_path = root / "review_checklist.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    expected_ids = [
        case_id
        for report in validation["splits"].values()
        for case_id in report["case_ids"]
    ]
    if Counter(row["case_id"] for row in reviews) != Counter(expected_ids):
        failures.append("review checklist does not cover every case exactly once")
    technical_yes = all(
        row.get(field) == "yes" for row in reviews for field in CHECK_FIELDS
    )
    if not technical_yes:
        failures.append("technical prereview fields are incomplete")
    signoff_path = root / "review_signoff.json"
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    signoff_pending = (
        not str(signoff.get("reviewer", "")).strip()
        and signoff.get("decision") == "pending_human_review"
        and signoff.get("review_checklist_sha256") == sha256_file(review_path)
        and signoff.get("reviewed_case_count") == validation["total_cases"]
    )
    if not signoff_pending:
        failures.append("signoff must remain pending during migration audit")

    return {
        "benchmark_version": "1.2-candidate",
        "audit_version": AUDIT_VERSION,
        "status": "passed_pending_human_review" if not failures else "failed",
        "valid": not failures,
        "case_count": validation["total_cases"],
        "budget_answer_spec_count": validation["budget_answer_specs"],
        "parent_equivalent_case_count": changed_case_count,
        "unauthorized_change_count": len(unauthorized_changes),
        "technical_review_rows": len(reviews),
        "technical_fields_all_yes": technical_yes,
        "human_signatures_present": int(bool(str(signoff.get("reviewer", "")).strip())),
        "parent_split_sha256": parent_hashes,
        "split_sha256": {
            split: report["sha256"] for split, report in validation["splits"].items()
        },
        "review_sha256": sha256_file(review_path),
        "scoring_contract": "quantity-semantic-v1",
        "failures": failures,
    }


def write_audit(data_dir: str | Path, parent_dir: str | Path) -> dict:
    root = Path(data_dir)
    report = audit_candidates(root, parent_dir)
    json_path = root / "deep_audit.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "deep_audit.md").write_text(
        "# Benchmark v1.2 Scoring-Migration Audit\n\n"
        f"Status: **{report['status']}**.\n\n"
        f"- Cases compared with frozen v1.1 parent: {report['parent_equivalent_case_count']}\n"
        f"- Unauthorized case changes: {report['unauthorized_change_count']}\n"
        f"- Typed Budget answer contracts: {report['budget_answer_spec_count']}\n"
        f"- Technical review rows: {report['technical_review_rows']}\n"
        f"- Human signatures present during audit: {report['human_signatures_present']}\n\n"
        "Only declared version metadata and Budget `answer_spec` fields differ from "
        "the frozen v1.1 parent. Human package signoff is still required.\n",
        encoding="utf-8",
    )
    signoff = write_signoff_template(
        root,
        root / "review_checklist.csv",
        deep_audit_path=json_path,
    )
    manifest_path = root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deep_audit"] = {
        "file": json_path.name,
        "sha256": sha256_file(json_path),
        "version": AUDIT_VERSION,
        "status": report["status"],
        "valid": report["valid"],
    }
    manifest["review_signoff"] = {
        "file": "review_signoff.json",
        "unsigned_template_sha256": sha256_file(root / "review_signoff.json"),
        "scope": signoff["scope"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit Benchmark v1.2 migration.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "benchmark/data/v1.2")
    parser.add_argument("--parent-dir", type=Path, default=project_root / "benchmark/data/v1.1")
    args = parser.parse_args()
    print(json.dumps(write_audit(args.data_dir, args.parent_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
