"""Freeze Benchmark v1.1 only after deep audit and complete human review."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import sha256_file
from .freeze_v1 import CHECK_FIELDS
from .validate_v11 import validate_candidate_splits


TOOL_FILES = (
    "generate_v11.py",
    "validate_v11.py",
    "prereview_v11.py",
    "audit_v11.py",
    "freeze_v11.py",
    "verify_external_v11.py",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_benchmark_v11(
    data_dir: str | Path,
    review_path: str | Path | None = None,
    signoff_path: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    validation = validate_candidate_splits(root)
    expected_ids = {
        case_id
        for split in validation["splits"].values()
        for case_id in split["case_ids"]
    }

    review_file = Path(review_path) if review_path else root / "review_checklist.csv"
    with review_file.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    review_by_id = {row.get("case_id", ""): row for row in reviews}
    if len(reviews) != len(expected_ids) or set(review_by_id) != expected_ids:
        raise ValueError("review checklist must contain every v1.1 case exactly once")

    technical_failures = []
    for case_id in sorted(expected_ids):
        row = review_by_id[case_id]
        technical_pass = all(
            row.get(field, "").strip().lower() == "yes" for field in CHECK_FIELDS
        )
        if not technical_pass:
            technical_failures.append(case_id)
    if technical_failures:
        raise ValueError(
            f"technical prereview incomplete for {len(technical_failures)} "
            f"v1.1 case(s): {technical_failures[:5]}"
        )

    audit_path = root / "deep_audit.json"
    audit_markdown_path = root / "deep_audit.md"
    if not audit_path.exists():
        raise ValueError("v1.1 deep audit artifact is missing")
    audit = _load_json(audit_path)
    if not audit.get("valid") or audit.get("status") != "passed_pending_human_review":
        raise ValueError("v1.1 deep audit did not pass")
    if audit.get("case_count") != len(expected_ids) or audit.get("failures"):
        raise ValueError("v1.1 deep audit case coverage or failure state is invalid")
    for split, split_report in validation["splits"].items():
        if audit.get("split_sha256", {}).get(split) != split_report["sha256"]:
            raise ValueError(f"v1.1 split changed after deep audit: {split}")

    signoff_file = Path(signoff_path) if signoff_path else root / "review_signoff.json"
    if not signoff_file.exists():
        raise ValueError("one-time human review signoff is missing")
    signoff = _load_json(signoff_file)
    reviewer = str(signoff.get("reviewer", "")).strip()
    approved = str(signoff.get("decision", "")).strip().lower() == "approved"
    if not reviewer or not approved:
        raise ValueError(
            "one-time human review signoff incomplete: fill reviewer and set decision=approved"
        )
    if signoff.get("scope") != "all_development_test_holdout_candidates":
        raise ValueError("human signoff scope does not cover all v1.1 candidate splits")
    if signoff.get("reviewed_case_count") != len(expected_ids):
        raise ValueError("human signoff case count does not match v1.1 candidates")
    if signoff.get("review_checklist_sha256") != sha256_file(review_file):
        raise ValueError("review checklist changed after one-time human signoff template")
    if signoff.get("deep_audit_sha256") != sha256_file(audit_path):
        raise ValueError("deep audit changed after one-time human signoff template")
    for split, split_report in validation["splits"].items():
        if signoff.get("split_sha256", {}).get(split) != split_report["sha256"]:
            raise ValueError(f"v1.1 split does not match one-time human signoff: {split}")

    candidate_path = root / "candidate_manifest.json"
    source_path = root / "source_manifest.json"
    candidate = _load_json(candidate_path)
    for split, split_report in validation["splits"].items():
        candidate_split = candidate.get("splits", {}).get(split, {})
        if candidate_split.get("sha256") != split_report["sha256"]:
            raise ValueError(f"candidate manifest does not match v1.1 split: {split}")

    benchmark_module = Path(__file__).resolve().parent
    tooling = {
        filename: sha256_file(benchmark_module / filename) for filename in TOOL_FILES
    }
    portable_splits = {
        split: {
            "file": f"{split}.jsonl",
            **{key: value for key, value in report.items() if key != "path"},
        }
        for split, report in validation["splits"].items()
    }
    manifest = {
        **validation,
        "benchmark_version": "1.1",
        "status": "frozen",
        "splits": portable_splits,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "review_file": review_file.name,
        "review_sha256": sha256_file(review_file),
        "reviewed_case_count": len(expected_ids),
        "reviewers": [reviewer],
        "review_signoff_file": signoff_file.name,
        "review_signoff_sha256": sha256_file(signoff_file),
        "deep_audit_file": audit_path.name,
        "deep_audit_sha256": sha256_file(audit_path),
        "deep_audit_report_file": audit_markdown_path.name,
        "deep_audit_report_sha256": sha256_file(audit_markdown_path),
        "deep_audit_version": audit["audit_version"],
        "candidate_manifest_file": candidate_path.name,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "source_manifest_file": source_path.name,
        "source_manifest_sha256": sha256_file(source_path),
        "tooling_sha256": tooling,
    }
    output = root / "frozen_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Benchmark v1.1 after deep audit and human review."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.1",
    )
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--signoff", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_benchmark_v11(args.data_dir, args.reviews, args.signoff),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
