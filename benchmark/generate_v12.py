"""Deterministically migrate frozen Benchmark v1.1 to v1.2 scoring candidates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from .artifacts import sha256_file
from .execution import require_frozen_benchmark
from .generate_v11 import DEFAULT_SEED
from .metrics import QUANTITY_ANSWER_SCORER_VERSION


GENERATOR_VERSION = "v1.2.0"
SPLITS = ("development", "test", "holdout")
REVIEW_FIELDS = (
    "case_id",
    "question_clear",
    "answer_correct",
    "evidence_ids_correct",
    "forbidden_ids_correct",
    "timestamps_correct",
    "aliases_checked",
    "answer_spec_checked",
    "notes",
)


def quantity_answer_spec(expected_answer: str) -> dict:
    return {
        "type": "quantity",
        "canonical_value": str(expected_answer),
        "value_aliases": [],
        "units": ["条"],
        "unit_policy": "optional",
        "output_format": "bare_value",
        "scorer_version": QUANTITY_ANSWER_SCORER_VERSION,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def upgrade_case(parent: dict) -> dict:
    """Apply the only permitted v1.1 -> v1.2 case-level changes."""
    case = copy.deepcopy(parent)
    metadata = case["metadata"]
    metadata.update({
        "benchmark_version": "1.2-candidate",
        "review_status": "pending_human_review",
        "generator_version": GENERATOR_VERSION,
        "author": "amr-v12-scoring-migration",
        "parent_benchmark_version": "1.1",
        "change_scope": "typed_budget_answer_semantics",
    })
    if case["category"] == "budget":
        case["answer_spec"] = quantity_answer_spec(case["expected_answer"])
    else:
        case.pop("answer_spec", None)
    return case


def build_all_splits(parent_dir: str | Path) -> dict[str, list[dict]]:
    root = Path(parent_dir)
    result = {}
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        require_frozen_benchmark(path)
        result[split] = [upgrade_case(case) for case in _read_jsonl(path)]
    return result


def write_all_splits(
    output_dir: str | Path,
    parent_dir: str | Path,
    *,
    scoring_protocol: str | Path,
) -> dict:
    root = Path(output_dir)
    parent_root = Path(parent_dir)
    protocol_path = Path(scoring_protocol)
    root.mkdir(parents=True, exist_ok=True)
    splits = build_all_splits(parent_root)
    split_manifest = {}
    all_cases = []
    for split, cases in splits.items():
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
        split_manifest[split] = {
            "file": path.name,
            "case_count": len(cases),
            "sha256": sha256_file(path),
        }
        all_cases.extend(cases)

    checklist_path = root / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in all_cases:
            writer.writerow({
                "case_id": case["case_id"],
                "notes": "pending-technical-prereview",
            })

    parent_manifest = parent_root / "frozen_manifest.json"
    lineage = {
        "migration_type": "scoring_contract_only",
        "parent_benchmark_version": "1.1",
        "parent_frozen_manifest_file": str(parent_manifest.resolve()),
        "parent_frozen_manifest_sha256": sha256_file(parent_manifest),
        "parent_split_sha256": {
            split: sha256_file(parent_root / f"{split}.jsonl") for split in SPLITS
        },
        "scoring_protocol_file": str(protocol_path.resolve()),
        "scoring_protocol_sha256": sha256_file(protocol_path),
        "allowed_case_changes": [
            "metadata benchmark/review/generator/author/lineage fields",
            "answer_spec on budget cases",
        ],
    }
    lineage_path = root / "lineage_manifest.json"
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "benchmark_version": "1.2-candidate",
        "generator_version": GENERATOR_VERSION,
        "seed": DEFAULT_SEED,
        "status": "pending_human_review",
        "total_case_count": len(all_cases),
        "budget_answer_spec_count": sum(case["category"] == "budget" for case in all_cases),
        "splits": split_manifest,
        "review_file": checklist_path.name,
        "review_sha256": sha256_file(checklist_path),
        "lineage_file": lineage_path.name,
        "lineage_sha256": sha256_file(lineage_path),
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate Benchmark v1.2 scoring candidates.")
    parser.add_argument("--output-dir", type=Path, default=project_root / "benchmark/data/v1.2")
    parser.add_argument("--parent-dir", type=Path, default=project_root / "benchmark/data/v1.1")
    parser.add_argument(
        "--scoring-protocol",
        type=Path,
        default=project_root / "docs/answer_scoring_protocol_v1.2.md",
    )
    args = parser.parse_args()
    print(json.dumps(
        write_all_splits(args.output_dir, args.parent_dir, scoring_protocol=args.scoring_protocol),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
