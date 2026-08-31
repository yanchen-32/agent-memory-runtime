from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable


QUESTION_TYPE_TO_CATEGORY = {
    "single-session-user": "fact_recall",
    "single-session-assistant": "fact_recall",
    "single-session-preference": "semantic_recall",
    "multi-session": "multi_hop",
    "temporal-reasoning": "temporal",
    "knowledge-update": "update",
}


def _timestamp(value: str) -> str:
    parsed = datetime.strptime(value, "%Y/%m/%d (%a) %H:%M")
    # LongMemEval timestamps have no timezone. UTC is a deterministic adapter
    # convention, not a claim about the original conversations' location.
    return parsed.replace(tzinfo=timezone.utc).isoformat()


def _session_content(date: str, messages: list[dict]) -> str:
    lines = [f"SESSION_TIME: {_timestamp(date)}"]
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def convert_case(payload: dict) -> dict:
    session_ids = list(payload["haystack_session_ids"])
    dates = list(payload["haystack_dates"])
    sessions = list(payload["haystack_sessions"])
    if not (len(session_ids) == len(dates) == len(sessions)):
        raise ValueError(
            f"unaligned LongMemEval sessions for {payload.get('question_id')}"
        )

    question_id = str(payload["question_id"])
    question_type = str(payload["question_type"])
    category = (
        "abstention"
        if question_id.endswith("_abs")
        else QUESTION_TYPE_TO_CATEGORY.get(question_type)
    )
    if category is None:
        raise ValueError(f"unsupported LongMemEval question type: {question_type}")

    totals = Counter(session_ids)
    occurrences: defaultdict[str, int] = defaultdict(int)
    conversation = []
    for session_id, date, session in zip(session_ids, dates, sessions):
        occurrences[session_id] += 1
        memory_id = (
            session_id
            if totals[session_id] == 1
            else f"{session_id}__occ{occurrences[session_id]}"
        )
        conversation.append({
            "memory_id": memory_id,
            "role": "user",
            "content": _session_content(date, session),
            "created_at": _timestamp(date),
            "valid_from": _timestamp(date),
            "metadata": {
                "source_dataset": "LongMemEval-S-Cleaned",
                "source_session_id": session_id,
                "source_session_occurrence": occurrences[session_id],
                "source_session_date": date,
            },
        })
    query_time = _timestamp(str(payload["question_date"]))
    expected_ids = list(payload.get("answer_session_ids") or [])
    if category != "abstention" and not expected_ids:
        raise ValueError(f"answer_session_ids missing for {question_id}")
    duplicated_answer_ids = set(expected_ids) & {
        session_id for session_id, count in totals.items() if count > 1
    }
    if duplicated_answer_ids:
        raise ValueError(
            f"duplicated answer session IDs need explicit adjudication for "
            f"{question_id}: {sorted(duplicated_answer_ids)}"
        )

    return {
        "case_id": f"longmemeval_{question_id}",
        "category": category,
        "conversation": conversation,
        "query": str(payload["question"]),
        "expected_memory_ids": expected_ids,
        "expected_answer": str(payload["answer"]),
        "expected_version": "",
        "query_time": query_time,
        "difficulty": "hard" if len(conversation) >= 40 else "medium",
        "answer_aliases": [],
        "token_budget": None,
        "memory_query_time": query_time,
        "forget_memory_ids": [],
        "memory_metadata": {},
        "forbidden_memory_ids": [],
        "metadata": {
            "source_dataset": "LongMemEval-S-Cleaned",
            "source_question_id": question_id,
            "source_question_type": question_type,
            "source_answer_session_ids": expected_ids,
            "source_license": "MIT",
            "source_split": "external_test",
            "session_count": len(conversation),
            "duplicate_source_session_ids": sorted(
                session_id for session_id, count in totals.items() if count > 1
            ),
            "timestamp_timezone_assumption": "UTC",
            "ground_truth_modified": False,
        },
    }


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    with Path(input_path).open("r", encoding="utf-8") as stream:
        source = json.load(stream)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        source = source[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for payload in source:
            stream.write(json.dumps(convert_case(payload), ensure_ascii=False) + "\n")
    return len(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official LongMemEval-S Cleaned to AMR JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_file(args.input, args.output, limit=args.limit)
    print(f"converted_cases={count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
