from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path


ALLOWED_CATEGORIES = {
    "fact_recall",
    "semantic_recall",
    "temporal",
    "update",
    "conflict",
    "long_context",
    "noise",
    "abstention",
    "budget",
    "multi_hop",
    "forgetting",
    "consolidation",
}


def _parse_datetime(value: str, field: str, case_id: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{case_id}: invalid {field}: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{case_id}: {field} must include timezone")


def validate_file(path: str | Path) -> dict[str, object]:
    seen_case_ids: set[str] = set()
    categories: Counter[str] = Counter()
    cases = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = str(payload.get("case_id", ""))
            if not case_id:
                raise ValueError(f"line {line_number}: missing case_id")
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen_case_ids.add(case_id)

            category = str(payload.get("category", ""))
            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"{case_id}: invalid category: {category}")
            categories[category] += 1

            conversation = payload.get("conversation") or []
            memory_ids = [str(turn.get("memory_id", "")) for turn in conversation]
            if not conversation or any(not memory_id for memory_id in memory_ids):
                raise ValueError(f"{case_id}: conversation/memory_id missing")
            if len(memory_ids) != len(set(memory_ids)):
                raise ValueError(f"{case_id}: duplicate conversation memory_id")

            expected = set(payload.get("expected_memory_ids") or [])
            forbidden = set(payload.get("forbidden_memory_ids") or [])
            available = set(memory_ids)
            if not expected <= available:
                raise ValueError(f"{case_id}: expected memory IDs absent from conversation")
            if not forbidden <= available:
                raise ValueError(f"{case_id}: forbidden memory IDs absent from conversation")
            if expected & forbidden:
                raise ValueError(f"{case_id}: expected and forbidden IDs overlap")
            if category != "abstention" and not str(payload.get("expected_answer", "")):
                raise ValueError(f"{case_id}: expected_answer missing")

            _parse_datetime(str(payload.get("query_time", "")), "query_time", case_id)
            memory_query_time = payload.get("memory_query_time")
            if memory_query_time:
                _parse_datetime(str(memory_query_time), "memory_query_time", case_id)
            for turn in conversation:
                for field in ("created_at", "valid_from"):
                    if turn.get(field):
                        _parse_datetime(str(turn[field]), field, case_id)
            cases += 1
    return {
        "path": str(Path(path).resolve()),
        "cases": cases,
        "categories": dict(sorted(categories.items())),
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AMR Benchmark v1 JSONL.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_file(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
