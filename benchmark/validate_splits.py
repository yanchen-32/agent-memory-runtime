from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

from memory import RuleMemoryExtractor

from .artifacts import sha256_file
from .loader import load_jsonl
from .validate_v1 import validate_file


SPLIT_NAMES = ("development", "test", "holdout")


def _point(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _validate_pair(pair_id: str, cases: list) -> None:
    modes = {case.metadata.get("query_mode"): case for case in cases}
    if set(modes) != {"current", "historical"} or len(cases) != 2:
        raise ValueError(f"{pair_id}: requires one current and one historical case")
    current = modes["current"]
    historical = modes["historical"]
    if current.conversation != historical.conversation:
        raise ValueError(f"{pair_id}: paired conversations differ")
    if current.category not in {"update", "conflict"}:
        raise ValueError(f"{pair_id}: current case must be update or conflict")
    if historical.category != "temporal":
        raise ValueError(f"{pair_id}: historical case must be temporal")

    turns = current.conversation
    if len(turns) != 2 or not all(turn.get("valid_from") for turn in turns):
        raise ValueError(f"{pair_id}: requires exactly two timestamped versions")
    old_id, new_id = (str(turn["memory_id"]) for turn in turns)
    old_time, new_time = (_point(str(turn["valid_from"])) for turn in turns)
    if old_time >= new_time:
        raise ValueError(f"{pair_id}: version timestamps are not increasing")
    if current.expected_memory_ids != [new_id] or current.forbidden_memory_ids != [old_id]:
        raise ValueError(f"{pair_id}: current expected/forbidden IDs are invalid")
    if historical.expected_memory_ids != [old_id] or historical.forbidden_memory_ids != [new_id]:
        raise ValueError(f"{pair_id}: historical expected/forbidden IDs are invalid")
    if _point(str(current.memory_query_time)) < new_time:
        raise ValueError(f"{pair_id}: current query precedes the new version")
    historical_time = _point(str(historical.memory_query_time))
    if not old_time <= historical_time < new_time:
        raise ValueError(f"{pair_id}: historical query is outside the old validity interval")

    extracted = RuleMemoryExtractor().extract(turns)
    if len(extracted) != 2:
        raise ValueError(f"{pair_id}: rule extractor did not produce two facts")
    old_fact, new_fact = extracted
    if not old_fact.subject or not old_fact.predicate:
        raise ValueError(f"{pair_id}: old version lacks a structured fact key")
    if (old_fact.subject, old_fact.predicate) != (new_fact.subject, new_fact.predicate):
        raise ValueError(f"{pair_id}: versions do not share subject + predicate")
    if old_fact.object_value == new_fact.object_value:
        raise ValueError(f"{pair_id}: version values are identical")


def validate_splits(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    all_case_ids: set[str] = set()
    families_by_split: dict[str, set[str]] = {}
    split_reports = {}
    pairs: defaultdict[str, list] = defaultdict(list)

    for split in SPLIT_NAMES:
        path = root / f"{split}.jsonl"
        report = validate_file(path)
        cases = load_jsonl(path)
        families: set[str] = set()
        for case in cases:
            if case.case_id in all_case_ids:
                raise ValueError(f"case ID leaks across splits: {case.case_id}")
            all_case_ids.add(case.case_id)
            if case.metadata.get("split") != split:
                raise ValueError(f"{case.case_id}: metadata split mismatch")
            family = str(case.metadata.get("scenario_family", ""))
            if not family:
                raise ValueError(f"{case.case_id}: scenario_family missing")
            families.add(family)
            pair_id = case.metadata.get("pair_id")
            if pair_id:
                pairs[str(pair_id)].append(case)
        families_by_split[split] = families
        split_reports[split] = {
            **report,
            "sha256": sha256_file(path),
            "case_ids": [case.case_id for case in cases],
            "scenario_families": sorted(families),
        }

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1:]:
            overlap = families_by_split[left] & families_by_split[right]
            if overlap:
                raise ValueError(f"scenario family leakage {left}/{right}: {sorted(overlap)}")
    for pair_id, cases in pairs.items():
        _validate_pair(pair_id, cases)

    return {
        "benchmark_version": "1.0-candidate",
        "status": "pending_human_review",
        "valid": True,
        "total_cases": len(all_case_ids),
        "governance_pairs": len(pairs),
        "splits": split_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Benchmark v1.0 split integrity.")
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "data" / "v1.0",
    )
    args = parser.parse_args()
    print(json.dumps(validate_splits(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
