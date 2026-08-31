from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceEvent:
    trace_id: str
    query_id: str | None
    stage: str
    memory_id: str | None = None
    version: int | None = None
    source_ids: list[str] = field(default_factory=list)
    score_components: dict[str, float | int | None] = field(default_factory=dict)
    rank: int | None = None
    selected: bool | None = None
    reject_reason: str | None = None
    token_cost: int | None = None
    latency_ms: float | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceRecorder:
    """In-process mechanistic trace sink; disabled mode stores nothing."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._events: dict[str, list[TraceEvent]] = {}

    def new_trace_id(self) -> str:
        return f"trace-{uuid4().hex}"

    def record(self, event: TraceEvent) -> None:
        if self.enabled:
            self._events.setdefault(event.trace_id, []).append(event)

    def get(self, trace_id: str) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events.get(trace_id, [])]

    def all(self) -> list[dict[str, Any]]:
        return [
            event.to_dict()
            for events in self._events.values()
            for event in events
        ]

    def clear(self) -> None:
        self._events.clear()
