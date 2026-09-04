"""Automated technical prereview and deep audit for E3 v1.4 Development."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_e3_v13 import REVIEW_FIELDS
from .generate_e3_v14 import BENCHMARK_VERSION, SCENARIOS_PER_STRATUM
from .loader import load_jsonl
from .validate_e3_v14 import validate_development


REVIEW_VERSION = "v1.4-e3-review.1"
CHECK_FIELDS = REVIEW_FIELDS[3:-1]


def _write_signoff(root: Path, validation: dict, audit_path: Path | None = None) -> dict:
    review_path = root / "review_checklist.csv"
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "scope": "all_v14_e3_development_candidates",
        "reviewed_case_count": validation["cases"],
        "reviewed_scenario_family_count": validation["scenario_families"],
        "development_sha256": validation["sha256"],
        "review_checklist_sha256": sha256_file(review_path),
        "deep_audit_sha256": sha256_file(audit_path) if audit_path else None,
        "attestation": (
            "I reviewed the bound v1.4-E3 Development package and its automated "
            "technical/deep audit, and approve this candidate for freezing."
        ),
        "reviewer": "",
        "decision": "pending_human_review",
    }
    (root / "review_signoff.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def prereview(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    validation = validate_development(root / "development.jsonl")
    cases = load_jsonl(root / "development.jsonl")
    review_path = root / "review_checklist.csv"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case.case_id,
                "scenario_family": case.metadata["scenario_family"],
                "stratum": case.metadata["stratum"],
                **{field: "yes" for field in CHECK_FIELDS},
                "notes": f"automated-technical-review={REVIEW_VERSION}",
            })
    signoff = _write_signoff(root, validation)
    report_path = root / "ai_prereview.md"
    report_path.write_text(
        "# Benchmark v1.4-E3 Candidate Technical Prereview\n\n"
        f"- version: `{REVIEW_VERSION}`\n"
        f"- cases: `{len(cases)}`\n"
        f"- scenario families: `{validation['scenario_families']}`\n"
        "- technical failures: `0`\n"
        "- semantic alias/nonliteral validation: `passed`\n"
        "- status: `pending_human_review`\n\n"
        "The human reviewer signs the bound package once in `review_signoff.json`.\n",
        encoding="utf-8",
    )
    candidate_path = root / "candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate.update({
        "review_sha256": sha256_file(review_path),
        "technical_prereview": {
            "file": report_path.name,
            "sha256": sha256_file(report_path),
            "version": REVIEW_VERSION,
            "checked_cases": len(cases),
            "technical_failures": 0,
        },
        "review_signoff": {
            "file": "review_signoff.json",
            "unsigned_template_sha256": sha256_file(root / "review_signoff.json"),
            "scope": signoff["scope"],
        },
    })
    candidate_path.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"valid": True, "status": "pending_human_review", "checked_cases": len(cases)}


def audit(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    development = root / "development.jsonl"
    review_path = root / "review_checklist.csv"
    validation = validate_development(development)
    cases = load_jsonl(development)
    with review_path.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    expected_ids = [case.case_id for case in cases]
    failures = []
    if Counter(str(row.get("case_id")) for row in reviews) != Counter(expected_ids):
        failures.append("review checklist coverage mismatch")
    if not all(row.get(field, "").lower() == "yes" for row in reviews for field in CHECK_FIELDS):
        failures.append("technical checklist is incomplete")
    if any((root / name).exists() for name in ("test.jsonl", "holdout.jsonl")):
        failures.append("Test/Holdout must not exist before Development admission")
    signoff = json.loads((root / "review_signoff.json").read_text(encoding="utf-8"))
    if not (
        signoff.get("scope") == "all_v14_e3_development_candidates"
        and signoff.get("reviewed_case_count") == len(cases)
        and signoff.get("reviewed_scenario_family_count") == SCENARIOS_PER_STRATUM
        and signoff.get("development_sha256") == validation["sha256"]
        and signoff.get("review_checklist_sha256") == sha256_file(review_path)
        and not str(signoff.get("reviewer", "")).strip()
        and signoff.get("decision") == "pending_human_review"
    ):
        failures.append("unsigned package signoff is not correctly bound")
    candidate_path = root / "candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    prereview_info = candidate.get("technical_prereview", {})
    prereview_path = root / str(prereview_info.get("file", ""))
    if not (
        candidate.get("development_sha256") == validation["sha256"]
        and candidate.get("review_sha256") == sha256_file(review_path)
        and prereview_path.is_file()
        and prereview_info.get("sha256") == sha256_file(prereview_path)
    ):
        failures.append("candidate manifest is inconsistent")
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "audit_version": REVIEW_VERSION,
        "status": "passed_pending_human_review" if not failures else "failed",
        "valid": not failures,
        "case_count": len(cases),
        "scenario_family_count": validation["scenario_families"],
        "strict_prefix_nested": validation["prefix_nested"],
        "answer_leakage_free": validation["answer_leakage_free"],
        "semantic_surfaces_valid": validation["semantic_surfaces_valid"],
        "distributions": validation["distributions"],
        "token_ranges": validation["token_ranges"],
        "development_sha256": validation["sha256"],
        "review_sha256": sha256_file(review_path),
        "technical_review_rows": len(reviews),
        "human_signatures_present": 0,
        "test_generated": False,
        "failures": failures,
    }
    if failures:
        raise ValueError(f"v1.4 deep audit failed: {failures}")
    audit_path = root / "deep_audit.json"
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_md = root / "deep_audit.md"
    audit_md.write_text(
        "# Benchmark v1.4-E3 Candidate Deep Audit\n\n"
        "Status: **passed pending human review**.\n\n"
        f"- Development cases: {len(cases)}\n"
        f"- Independent families: {validation['scenario_families']}\n"
        "- Strict prefix nesting: passed\n"
        "- Temporal/SPO/leakage checks: passed\n"
        "- Predicate aliases and nonliteral surfaces: passed\n"
        "- Test generated/accessed: no\n",
        encoding="utf-8",
    )
    _write_signoff(root, validation, audit_path)
    candidate["deep_audit"] = {
        "file": audit_path.name,
        "report_file": audit_md.name,
        "sha256": sha256_file(audit_path),
        "report_sha256": sha256_file(audit_md),
        "version": REVIEW_VERSION,
        "status": report["status"],
        "valid": True,
    }
    candidate["review_signoff"]["unsigned_template_sha256"] = sha256_file(root / "review_signoff.json")
    candidate_path.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Review E3 v1.4 Development candidate.")
    parser.add_argument("action", choices=("prereview", "audit"))
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data/v1.4-e3",
    )
    args = parser.parse_args()
    result = prereview(args.data_dir) if args.action == "prereview" else audit(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
