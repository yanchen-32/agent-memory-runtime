from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


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
    answer_aliases: list[str] = field(default_factory=list)
    token_budget: int | None = None
    memory_query_time: str | None = None
    forget_memory_ids: list[str] = field(default_factory=list)
    memory_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    forbidden_memory_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(BenchmarkCase(**payload))
    return cases
