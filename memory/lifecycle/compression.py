from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from memory.schema import MemoryRecord


@dataclass(slots=True)
class CompressionResult:
    text: str
    original_chars: int
    compressed_chars: int
    compressed: bool

    @property
    def ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return 1.0 - self.compressed_chars / self.original_chars


class MemoryCompressor:
    """Single-memory V1 compressor with optional LLM summarizer hook."""

    def __init__(self, max_chars: int = 240, summarizer: Callable[[str, int], str] | None = None):
        self.max_chars = max_chars
        self.summarizer = summarizer

    def compress(self, memory: MemoryRecord | str) -> CompressionResult:
        text = memory.content if isinstance(memory, MemoryRecord) else str(memory)
        if len(text) <= self.max_chars:
            return CompressionResult(text, len(text), len(text), False)
        if self.summarizer is not None:
            summary = self.summarizer(text, self.max_chars).strip()
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])", text) if s.strip()]
            selected: list[str] = []
            length = 0
            for sentence in sentences:
                if length + len(sentence) > self.max_chars:
                    break
                selected.append(sentence)
                length += len(sentence)
            summary = "".join(selected) or text[: self.max_chars]
        summary = summary[: self.max_chars]
        return CompressionResult(summary, len(text), len(summary), len(summary) < len(text))
