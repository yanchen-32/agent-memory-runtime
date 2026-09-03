"""Freeze v1.3 E3 Development after one bound human package signoff."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_e3_v13 import SCENARIOS_PER_STRATUM
from .prereview_e3_v13 import CHECK_FIELDS
from .validate_e3_v13 import validate_development


TOOL_FILES = (
    "generate_e3_v13.py",
    "validate_e3_v13.py",
    "prereview_e3_v13.py",
    "audit_e3_v13.py",
    "freeze_e3_v13.py",
    "loader.py",
    "metrics.py",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_benchmark_e3_v13(
    data_dir: str | Path,
    review_path: str | Path | None = None,
    signoff_path: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    development_path = root / "development.jsonl"
    validation = validate_development(development_path)
    expected_ids = set(validation["case_ids"])

    review_file = Path(review_path) if review_path else root / "review_checklist.csv"
    with review_file.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    review_by_id = {str(row.get("case_id", "")): row for row in reviews}
    if len(reviews) != len(expected_ids) or set(review_by_id) != expected_ids:
        raise ValueError("review checklist must contain every v1.3 E3 case exactly once")
    incomplete = [
        case_id
        for case_id, row in review_by_id.items()
        if not all(
            row.get(field, "").strip().lower() == "yes" for field in CHECK_FIELDS
        )
    ]
    if incomplete:
        raise ValueError(f"technical prereview incomplete: {incomplete[:5]}")

    audit_path = root / "deep_audit.json"
    audit_report_path = root / "deep_audit.md"
    audit = _load_json(audit_path)
    if (
        not audit.get("valid")
        or audit.get("status") != "passed_pending_human_review"
        or audit.get("case_count") != len(expected_ids)
        or audit.get("scenario_family_count") != SCENARIOS_PER_STRATUM
        or audit.get("failures")
    ):
        raise ValueError("v1.3 E3 deep audit did not pass")
    if audit.get("development_sha256") != validation["sha256"]:
        raise ValueError("v1.3 E3 Development changed after deep audit")
    if audit.get("review_sha256") != sha256_file(review_file):
        raise ValueError("v1.3 E3 review checklist changed after deep audit")

    signoff_file = Path(signoff_path) if signoff_path else root / "review_signoff.json"
    signoff = _load_json(signoff_file)
    reviewer = str(signoff.get("reviewer", "")).strip()
    if not reviewer or str(signoff.get("decision", "")).strip().lower() != "approved":
        raise ValueError(
            "one-time human review signoff incomplete: fill reviewer and set decision=approved"
        )
    if signoff.get("scope") != "all_v13_e3_development_candidates":
        raise ValueError("human signoff does not cover all v1.3 E3 candidates")
    if signoff.get("reviewed_case_count") != len(expected_ids):
        raise ValueError("human signoff case count mismatch")
    if signoff.get("reviewed_scenario_family_count") != SCENARIOS_PER_STRATUM:
        raise ValueError("human signoff scenario-family count mismatch")
    if signoff.get("development_sha256") != validation["sha256"]:
        raise ValueError("Development candidate does not match human signoff")
    if signoff.get("review_checklist_sha256") != sha256_file(review_file):
        raise ValueError("review checklist does not match human signoff")
    if signoff.get("deep_audit_sha256") != sha256_file(audit_path):
        raise ValueError("deep audit does not match human signoff")

    candidate_path = root / "candidate_manifest.json"
    candidate = _load_json(candidate_path)
    if candidate.get("development_sha256") != validation["sha256"]:
        raise ValueError("candidate manifest Development hash mismatch")
    if candidate.get("review_sha256") != sha256_file(review_file):
        raise ValueError("candidate manifest review hash mismatch")
    if candidate.get("deep_audit", {}).get("sha256") != sha256_file(audit_path):
        raise ValueError("candidate manifest deep-audit hash mismatch")
    prereview = candidate.get("technical_prereview", {})
    prereview_path = root / str(prereview.get("file", ""))
    if not prereview_path.is_file() or prereview.get("sha256") != sha256_file(
        prereview_path
    ):
        raise ValueError("candidate manifest technical-prereview hash mismatch")
    if candidate.get("deep_audit", {}).get("report_sha256") != sha256_file(
        audit_report_path
    ):
        raise ValueError("candidate manifest deep-audit report hash mismatch")
    if candidate.get("review_signoff", {}).get("unsigned_template_sha256") == sha256_file(
        signoff_file
    ):
        raise ValueError("human signoff file is still the unsigned template")

    module_root = Path(__file__).resolve().parent
    tooling = {filename: sha256_file(module_root / filename) for filename in TOOL_FILES}
    runner_path = module_root.parent / "experiments" / "run_e3_scaling.py"
    tooling["experiments/run_e3_scaling.py"] = sha256_file(runner_path)
    manifest = {
        "benchmark_version": "1.3-e3",
        "status": "frozen",
        "splits": {
            "development": {
                "file": development_path.name,
                "sha256": validation["sha256"],
                "cases": len(expected_ids),
                "case_ids": validation["case_ids"],
            }
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scenario_family_count": validation["scenario_families"],
        "strata": validation["strata"],
        "token_ranges": validation["token_ranges"],
        "predicate_distribution": validation["predicate_distribution"],
        "query_template_distribution": validation["query_template_distribution"],
        "target_position_distribution": validation["target_position_distribution"],
        "distractor_kinds": validation["distractor_kinds"],
        "case_order_policy": validation["case_order_policy"],
        "strict_prefix_nested": validation["prefix_nested"],
        "answer_leakage_free": validation["answer_leakage_free"],
        "review_file": review_file.name,
        "review_sha256": sha256_file(review_file),
        "reviewed_case_count": len(expected_ids),
        "reviewers": [reviewer],
        "review_signoff_file": signoff_file.name,
        "review_signoff_sha256": sha256_file(signoff_file),
        "deep_audit_file": audit_path.name,
        "deep_audit_sha256": sha256_file(audit_path),
        "deep_audit_report_file": audit_report_path.name,
        "deep_audit_report_sha256": sha256_file(audit_report_path),
        "deep_audit_version": audit["audit_version"],
        "candidate_manifest_file": candidate_path.name,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "tooling_sha256": tooling,
        "test_generated": False,
        "holdout_generated": False,
    }
    output = root / "frozen_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze reviewed v1.3 E3 Development.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.3-e3",
    )
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--signoff", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(
        freeze_benchmark_e3_v13(args.data_dir, args.reviews, args.signoff),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
