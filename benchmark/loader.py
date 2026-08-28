from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    category: str
    conversation: list[dict]
    query: str
    expected_memory_ids: list[str]
    expected_answer: str
    expected_version: str
    query_time: str
    difficulty: str


def load_jsonl(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cases.append(BenchmarkCase(**json.loads(line)))
    return cases
