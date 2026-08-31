from __future__ import annotations

from abc import ABC, abstractmethod
import re
import time
from typing import Protocol


ANSWER_FORMAT_VERSION = "shortest-answer-v2-temporal"
ANSWER_FORMAT_INSTRUCTION = (
    "Reply with only the shortest answer, or UNKNOWN if unsupported.\n"
)
TEMPORAL_CONTEXT_INSTRUCTION = (
    "For this historical question, answer at QUERY_TIME. "
    "A memory applies over [VALID_FROM, VALID_TO); "
    "VALID_TO=UNSPECIFIED means the source did not state an end time.\n"
)


def _format_timestamp(value: object | None, fallback: str) -> str:
    if value is None:
        return fallback
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def temporal_header(query_time: object | None, enabled: bool) -> str:
    """Return the shared, explicit point-in-time interpretation contract."""
    if not enabled:
        return ""
    return (
        TEMPORAL_CONTEXT_INSTRUCTION
        + f"QUERY_TIME[{_format_timestamp(query_time, 'UNSPECIFIED')}]\n"
    )


def memory_line(
    index: int,
    content: str,
    *,
    temporal_context: bool = False,
    valid_from: object | None = None,
    valid_to: object | None = None,
) -> str:
    """Serialize evidence consistently without changing retrieval behavior."""
    if not temporal_context:
        return f"MEMORY[{index}] {content}"
    return (
        f"MEMORY[{index}] "
        f"VALID_FROM[{_format_timestamp(valid_from, 'UNSPECIFIED')}] "
        f"VALID_TO[{_format_timestamp(valid_to, 'UNSPECIFIED')}] "
        f"{content}"
    )


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate one response from a fully constructed prompt."""


class Agent(ABC):
    @abstractmethod
    def answer(self, query: str) -> str:
        raise NotImplementedError


def timed_generate(client: LLMClient, prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.generate(prompt)
    return response, (time.perf_counter() - started) * 1000


def estimate_tokens(text: str) -> int:
    """Deterministic tokenizer estimate for comparable offline experiments."""
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text))


def select_context_indices(
    lines: list[str],
    question: str,
    token_budget: int | None,
    header: str,
) -> list[int]:
    """Keep ranked/history lines while respecting a total prompt budget."""
    if token_budget is None:
        return list(range(len(lines)))
    if token_budget <= 0:
        return []

    selected: list[int] = []
    for index, line in enumerate(lines):
        candidate_lines = [lines[i] for i in selected] + [line]
        prompt = (
            header
            + "\n".join(candidate_lines)
            + "\n\n"
            + f"QUESTION: {question}\n"
        )
        if estimate_tokens(prompt) <= token_budget:
            selected.append(index)
        else:
            break
    return selected
