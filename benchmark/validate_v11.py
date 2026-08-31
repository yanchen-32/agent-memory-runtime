"""Structural validation for the project-owned AMR-CN Benchmark v1.1 candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

from memory import RuleMemoryExtractor

from .artifacts import sha256_file
from .loader import BenchmarkCase, load_jsonl
from .validate_v1 import validate_file


V11_VERSION = "1.1-candidate"
SPLIT_CASE_COUNTS = {
    "development": 288,
    "test": 96,
    "holdout": 96,
}
DEVELOPMENT_CATEGORIES = Counter({
    "update": 24,
    "conflict": 24,
    "temporal": 48,
    "fact_recall": 24,
    "semantic_recall": 24,
    "multi_hop": 24,
    "abstention": 24,
    "noise": 24,
    "budget": 24,
    "forgetting": 24,
    "long_context": 24,
})
EVALUATION_CATEGORIES = Counter({
    "update": 8,
    "conflict": 8,
    "temporal": 16,
    "fact_recall": 8,
    "semantic_recall": 8,
    "multi_hop": 8,
    "abstention": 8,
    "noise": 8,
    "budget": 8,
    "forgetting": 8,
    "long_context": 8,
})
SPLIT_CATEGORIES = {
    "development": DEVELOPMENT_CATEGORIES,
    "test": EVALUATION_CATEGORIES,
    "holdout": EVALUATION_CATEGORIES,
}
SPLIT_PAIR_COUNTS = {"development": 48, "test": 16, "holdout": 16}
REQUIRED_METADATA = {
    "benchmark_version",
    "source_type",
    "derivation_type",
    "split",
    "scenario_family",
    "author",
    "review_status",
    "generator_version",
}


def _point(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _check_pair(pair_id: str, pair: list[BenchmarkCase]) -> None:
    modes = {str(case.metadata.get("query_mode")): case for case in pair}
    if len(pair) != 2 or set(modes) != {"current", "historical"}:
        raise ValueError(f"{pair_id}: requires exactly one current and one historical case")
    current, historical = modes["current"], modes["historical"]
    if current.category not in {"update", "conflict"} or historical.category != "temporal":
        raise ValueError(f"{pair_id}: invalid current/historical categories")
    if current.conversation != historical.conversation:
        raise ValueError(f"{pair_id}: paired conversations differ")
    if len(current.conversation) != 2:
        raise ValueError(f"{pair_id}: requires exactly two memory versions")

    old, new = current.conversation
    old_id, new_id = str(old["memory_id"]), str(new["memory_id"])
    old_from, new_from = _point(str(old["valid_from"])), _point(str(new["valid_from"]))
    if old_from >= new_from:
        raise ValueError(f"{pair_id}: version valid_from values are not increasing")
    if old.get("valid_to") != str(new["valid_from"]):
        raise ValueError(f"{pair_id}: old version must end at the new valid_from")
    if new.get("valid_to") is not None:
        raise ValueError(f"{pair_id}: newest version must have unspecified valid_to")
    if current.expected_memory_ids != [new_id] or current.forbidden_memory_ids != [old_id]:
        raise ValueError(f"{pair_id}: invalid current expected/forbidden memory IDs")
    if historical.expected_memory_ids != [old_id] or historical.forbidden_memory_ids != [new_id]:
        raise ValueError(f"{pair_id}: invalid historical expected/forbidden memory IDs")
    if _point(str(current.memory_query_time)) < new_from:
        raise ValueError(f"{pair_id}: current query precedes latest version")
    historical_time = _point(str(historical.memory_query_time))
    if not old_from <= historical_time < new_from:
        raise ValueError(f"{pair_id}: historical query falls outside old version interval")

    extracted = RuleMemoryExtractor().extract(current.conversation)
    if len(extracted) != 2:
        raise ValueError(f"{pair_id}: extractor did not see two version facts")
    old_fact, new_fact = extracted
    actual_key = (old_fact.subject, old_fact.predicate)
    if not all(actual_key) or actual_key != (new_fact.subject, new_fact.predicate):
        raise ValueError(f"{pair_id}: versions lack the same extractor-visible subject + predicate")
    declared_key = (current.metadata.get("subject"), current.metadata.get("predicate"))
    if declared_key != actual_key or declared_key != (
        historical.metadata.get("subject"), historical.metadata.get("predicate")
    ):
        raise ValueError(f"{pair_id}: declared and extractor-visible fact keys differ")
    if old_fact.object_value == new_fact.object_value:
        raise ValueError(f"{pair_id}: version values are identical")


def validate_split(path: str | Path, split: str) -> dict:
    """Validate one deterministic v1.1 candidate split."""
    if split not in SPLIT_CASE_COUNTS:
        raise ValueError(f"unsupported split: {split}")
    target = Path(path)
    base_report = validate_file(target)
    cases = load_jsonl(target)
    expected_cases = SPLIT_CASE_COUNTS[split]
    if len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} {split} cases, got {len(cases)}")
    categories = Counter(case.category for case in cases)
    if categories != SPLIT_CATEGORIES[split]:
        raise ValueError(f"unexpected {split} category distribution: {dict(categories)}")

    families: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    pairs: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    seen_content: set[tuple[str, ...]] = set()
    for case in cases:
        if set(case.metadata) < REQUIRED_METADATA:
            raise ValueError(f"{case.case_id}: missing v1.1 provenance metadata")
        if case.metadata["benchmark_version"] != V11_VERSION:
            raise ValueError(f"{case.case_id}: wrong benchmark version")
        if case.metadata["source_type"] != "project_owned":
            raise ValueError(f"{case.case_id}: v1.1 Development must be project-owned")
        if case.metadata["derivation_type"] != "independent_authoring_from_capability_spec":
            raise ValueError(f"{case.case_id}: invalid derivation policy")
        if case.metadata["split"] != split:
            raise ValueError(f"{case.case_id}: split metadata mismatch")
        if case.metadata["review_status"] != "pending_human_review":
            raise ValueError(f"{case.case_id}: candidate must remain pending human review")
        family = str(case.metadata["scenario_family"])
        families[family].append(case)
        pair_id = case.metadata.get("pair_id")
        if pair_id:
            pairs[str(pair_id)].append(case)
        elif len(families[family]) > 1:
            raise ValueError(f"{case.case_id}: non-paired scenario family is reused")
        content_key = tuple(str(turn["content"]) for turn in case.conversation)
        if content_key in seen_content and not pair_id:
            raise ValueError(f"{case.case_id}: duplicate non-paired conversation")
        seen_content.add(content_key)
        if case.category == "budget" and case.token_budget != 80:
            raise ValueError(f"{case.case_id}: budget case must use 80 tokens")
        if case.category == "forgetting":
            if not case.forget_memory_ids or set(case.forget_memory_ids) != set(case.forbidden_memory_ids):
                raise ValueError(f"{case.case_id}: forgetting policy must forbid forgotten memories")

    for pair_id, pair in pairs.items():
        _check_pair(pair_id, pair)
    if len(pairs) != SPLIT_PAIR_COUNTS[split]:
        raise ValueError(f"expected {SPLIT_PAIR_COUNTS[split]} governance pairs, got {len(pairs)}")
    for family, family_cases in families.items():
        pair_ids = {case.metadata.get("pair_id") for case in family_cases}
        if len(family_cases) > 1 and (len(family_cases) != 2 or len(pair_ids) != 1 or None in pair_ids):
            raise ValueError(f"{family}: scenario family leakage within Development")

    return {
        **base_report,
        "benchmark_version": V11_VERSION,
        "status": "pending_human_review",
        "governance_pairs": len(pairs),
        "scenario_families": len(families),
        "case_ids": [case.case_id for case in cases],
        "sha256": sha256_file(target),
    }


def validate_development(path: str | Path) -> dict:
    """Backward-compatible Development-only validator entry point."""
    return validate_split(path, "development")


def validate_candidate_splits(data_dir: str | Path) -> dict:
    """Validate all v1.1 candidates and reject scenario-family leakage."""
    root = Path(data_dir)
    reports = {}
    families_by_split = {}
    for split in SPLIT_CASE_COUNTS:
        path = root / f"{split}.jsonl"
        reports[split] = validate_split(path, split)
        families_by_split[split] = set(reports[split]["scenario_families"] and [
            case.metadata["scenario_family"] for case in load_jsonl(path)
        ])
    split_names = tuple(SPLIT_CASE_COUNTS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = families_by_split[left] & families_by_split[right]
            if overlap:
                raise ValueError(f"scenario family leakage {left}/{right}: {sorted(overlap)}")
    return {
        "benchmark_version": V11_VERSION,
        "status": "pending_human_review",
        "valid": True,
        "total_cases": sum(report["cases"] for report in reports.values()),
        "governance_pairs": sum(report["governance_pairs"] for report in reports.values()),
        "splits": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AMR-CN Benchmark v1.1 candidates.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "data" / "v1.1",
    )
    args = parser.parse_args()
    if args.path.is_dir():
        result = validate_candidate_splits(args.path)
    else:
        result = validate_development(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
