"""Structural and scoring-contract validation for Benchmark v1.2 candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_v12 import quantity_answer_spec
from .loader import BenchmarkCase, load_jsonl
from .validate_v1 import validate_file
from .validate_v11 import (
    SPLIT_CASE_COUNTS,
    SPLIT_CATEGORIES,
    SPLIT_PAIR_COUNTS,
    _check_pair,
)


V12_VERSION = "1.2-candidate"
SPLITS = tuple(SPLIT_CASE_COUNTS)
REQUIRED_METADATA = {
    "benchmark_version",
    "source_type",
    "derivation_type",
    "split",
    "scenario_family",
    "author",
    "review_status",
    "generator_version",
    "parent_benchmark_version",
    "change_scope",
}


def validate_split(path: str | Path, split: str) -> dict:
    if split not in SPLIT_CASE_COUNTS:
        raise ValueError(f"unsupported split: {split}")
    target = Path(path)
    base_report = validate_file(target)
    cases = load_jsonl(target)
    if len(cases) != SPLIT_CASE_COUNTS[split]:
        raise ValueError(f"unexpected {split} case count: {len(cases)}")
    categories = Counter(case.category for case in cases)
    if categories != SPLIT_CATEGORIES[split]:
        raise ValueError(f"unexpected {split} category distribution: {dict(categories)}")

    families: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    pairs: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    budget_specs = 0
    for case in cases:
        if set(case.metadata) < REQUIRED_METADATA:
            raise ValueError(f"{case.case_id}: missing v1.2 provenance metadata")
        if case.metadata["benchmark_version"] != V12_VERSION:
            raise ValueError(f"{case.case_id}: wrong benchmark version")
        if case.metadata["parent_benchmark_version"] != "1.1":
            raise ValueError(f"{case.case_id}: wrong parent benchmark version")
        if case.metadata["change_scope"] != "typed_budget_answer_semantics":
            raise ValueError(f"{case.case_id}: wrong migration scope")
        if case.metadata["review_status"] != "pending_human_review":
            raise ValueError(f"{case.case_id}: candidate review status must be pending")
        if case.metadata["split"] != split:
            raise ValueError(f"{case.case_id}: split metadata mismatch")
        if case.category == "budget":
            if case.token_budget != 80:
                raise ValueError(f"{case.case_id}: Budget ceiling must be 80")
            if case.answer_spec != quantity_answer_spec(case.expected_answer):
                raise ValueError(f"{case.case_id}: invalid or incomplete answer_spec")
            budget_specs += 1
        elif case.answer_spec is not None:
            raise ValueError(f"{case.case_id}: answer_spec is only enabled for Budget v1.2")

        family = str(case.metadata["scenario_family"])
        families[family].append(case)
        pair_id = case.metadata.get("pair_id")
        if pair_id:
            pairs[str(pair_id)].append(case)

    for pair_id, pair in pairs.items():
        _check_pair(pair_id, pair)
    if len(pairs) != SPLIT_PAIR_COUNTS[split]:
        raise ValueError(f"unexpected {split} governance pair count: {len(pairs)}")
    for family, family_cases in families.items():
        pair_ids = {case.metadata.get("pair_id") for case in family_cases}
        if len(family_cases) > 1 and (
            len(family_cases) != 2 or len(pair_ids) != 1 or None in pair_ids
        ):
            raise ValueError(f"{family}: invalid scenario-family reuse")
    expected_budget_specs = SPLIT_CATEGORIES[split]["budget"]
    if budget_specs != expected_budget_specs:
        raise ValueError(f"expected {expected_budget_specs} answer specs, got {budget_specs}")
    return {
        **base_report,
        "benchmark_version": V12_VERSION,
        "status": "pending_human_review",
        "governance_pairs": len(pairs),
        "scenario_families": len(families),
        "budget_answer_specs": budget_specs,
        "case_ids": [case.case_id for case in cases],
        "sha256": sha256_file(target),
    }


def validate_candidate_splits(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    reports = {
        split: validate_split(root / f"{split}.jsonl", split) for split in SPLITS
    }
    families = {
        split: {case.metadata["scenario_family"] for case in load_jsonl(root / f"{split}.jsonl")}
        for split in SPLITS
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = families[left] & families[right]
            if overlap:
                raise ValueError(f"scenario family leakage {left}/{right}: {sorted(overlap)}")
    all_ids = [case_id for report in reports.values() for case_id in report["case_ids"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("case IDs must be globally unique")
    return {
        "benchmark_version": V12_VERSION,
        "status": "pending_human_review",
        "valid": True,
        "total_cases": sum(report["cases"] for report in reports.values()),
        "budget_answer_specs": sum(report["budget_answer_specs"] for report in reports.values()),
        "governance_pairs": sum(report["governance_pairs"] for report in reports.values()),
        "splits": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Benchmark v1.2 candidates.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "data/v1.2",
    )
    args = parser.parse_args()
    result = (
        validate_candidate_splits(args.path)
        if args.path.is_dir()
        else validate_split(args.path, args.path.stem)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
