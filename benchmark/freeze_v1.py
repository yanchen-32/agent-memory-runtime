from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .validate_splits import validate_splits


CHECK_FIELDS = (
    "question_clear",
    "answer_correct",
    "evidence_ids_correct",
    "forbidden_ids_correct",
    "timestamps_correct",
    "aliases_checked",
)


def freeze_benchmark(data_dir: str | Path, review_path: str | Path) -> dict:
    root = Path(data_dir)
    validation = validate_splits(root)
    expected_ids = {
        case_id
        for split in validation["splits"].values()
        for case_id in split["case_ids"]
    }
    with Path(review_path).open("r", encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    review_by_id = {row.get("case_id", ""): row for row in reviews}
    if set(review_by_id) != expected_ids or len(reviews) != len(expected_ids):
        raise ValueError("review checklist must contain every case exactly once")

    pending = []
    for case_id in sorted(expected_ids):
        row = review_by_id[case_id]
        checks_pass = all(row.get(field, "").strip().lower() == "yes" for field in CHECK_FIELDS)
        approved = row.get("decision", "").strip().lower() == "approved"
        reviewer = row.get("reviewer", "").strip()
        if not (checks_pass and approved and reviewer):
            pending.append(case_id)
    if pending:
        raise ValueError(
            f"human review incomplete for {len(pending)} case(s): {pending[:5]}"
        )

    manifest = {
        **validation,
        "benchmark_version": "1.0",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "review_file": str(Path(review_path).name),
        "reviewed_case_count": len(expected_ids),
    }
    output = root / "frozen_manifest.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Benchmark v1.0 only after every human review check passes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.0",
    )
    parser.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_benchmark(args.data_dir, args.reviews),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
