"""Freeze Benchmark v1.2 after migration audit and one human signoff."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import sha256_file
from .execution import require_frozen_benchmark
from .prereview_v12 import CHECK_FIELDS
from .validate_v12 import SPLITS, validate_candidate_splits


TOOL_FILES = (
    "generate_v12.py",
    "validate_v12.py",
    "prereview_v12.py",
    "audit_v12.py",
    "freeze_v12.py",
    "metrics.py",
    "loader.py",
    "runner.py",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_benchmark_v12(
    data_dir: str | Path,
    parent_dir: str | Path,
    review_path: str | Path | None = None,
    signoff_path: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    parent_root = Path(parent_dir)
    validation = validate_candidate_splits(root)
    expected_ids = {
        case_id
        for split in validation["splits"].values()
        for case_id in split["case_ids"]
    }
    for split in SPLITS:
        require_frozen_benchmark(parent_root / f"{split}.jsonl")

    review_file = Path(review_path) if review_path else root / "review_checklist.csv"
    with review_file.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    review_by_id = {row.get("case_id", ""): row for row in reviews}
    if len(reviews) != len(expected_ids) or set(review_by_id) != expected_ids:
        raise ValueError("review checklist must contain every v1.2 case exactly once")
    incomplete = [
        case_id
        for case_id, row in review_by_id.items()
        if not all(row.get(field, "").strip().lower() == "yes" for field in CHECK_FIELDS)
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
        or audit.get("failures")
    ):
        raise ValueError("v1.2 migration audit did not pass")
    for split, report in validation["splits"].items():
        if audit.get("split_sha256", {}).get(split) != report["sha256"]:
            raise ValueError(f"v1.2 split changed after audit: {split}")
        parent_hash = sha256_file(parent_root / f"{split}.jsonl")
        if audit.get("parent_split_sha256", {}).get(split) != parent_hash:
            raise ValueError(f"v1.1 parent split changed after audit: {split}")

    signoff_file = Path(signoff_path) if signoff_path else root / "review_signoff.json"
    signoff = _load_json(signoff_file)
    reviewer = str(signoff.get("reviewer", "")).strip()
    if not reviewer or str(signoff.get("decision", "")).strip().lower() != "approved":
        raise ValueError(
            "one-time human review signoff incomplete: fill reviewer and set decision=approved"
        )
    if signoff.get("scope") != "all_development_test_holdout_scoring_candidates":
        raise ValueError("human signoff does not cover all v1.2 scoring candidates")
    if signoff.get("reviewed_case_count") != len(expected_ids):
        raise ValueError("human signoff case count mismatch")
    if signoff.get("review_checklist_sha256") != sha256_file(review_file):
        raise ValueError("review checklist changed after human signoff template")
    if signoff.get("deep_audit_sha256") != sha256_file(audit_path):
        raise ValueError("deep audit changed after human signoff template")
    for split, report in validation["splits"].items():
        if signoff.get("split_sha256", {}).get(split) != report["sha256"]:
            raise ValueError(f"v1.2 split does not match human signoff: {split}")

    candidate_path = root / "candidate_manifest.json"
    lineage_path = root / "lineage_manifest.json"
    candidate = _load_json(candidate_path)
    for split, report in validation["splits"].items():
        if candidate.get("splits", {}).get(split, {}).get("sha256") != report["sha256"]:
            raise ValueError(f"candidate manifest does not match v1.2 split: {split}")
    if candidate.get("lineage_sha256") != sha256_file(lineage_path):
        raise ValueError("candidate lineage hash mismatch")
    lineage = _load_json(lineage_path)
    parent_manifest_path = parent_root / "frozen_manifest.json"
    if lineage.get("parent_frozen_manifest_sha256") != sha256_file(parent_manifest_path):
        raise ValueError("v1.1 frozen parent manifest changed after generation")

    module_root = Path(__file__).resolve().parent
    tooling = {filename: sha256_file(module_root / filename) for filename in TOOL_FILES}
    manifest = {
        **validation,
        "benchmark_version": "1.2",
        "status": "frozen",
        "splits": {
            split: {
                "file": f"{split}.jsonl",
                **{key: value for key, value in report.items() if key != "path"},
            }
            for split, report in validation["splits"].items()
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "answer_scorer_version": "quantity-semantic-v1",
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
        "lineage_manifest_file": lineage_path.name,
        "lineage_manifest_sha256": sha256_file(lineage_path),
        "parent_frozen_manifest_sha256": sha256_file(parent_manifest_path),
        "tooling_sha256": tooling,
    }
    output = root / "frozen_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Freeze reviewed Benchmark v1.2.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "benchmark/data/v1.2")
    parser.add_argument("--parent-dir", type=Path, default=project_root / "benchmark/data/v1.1")
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--signoff", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(
        freeze_benchmark_v12(args.data_dir, args.parent_dir, args.reviews, args.signoff),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
