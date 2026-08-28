from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class MemoryAction(str, Enum):
    ADD = "ADD"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    IGNORE = "IGNORE"
    ARCHIVE = "ARCHIVE"


class ConflictType(str, Enum):
    NONE = "none"
    VALUE_CONFLICT = "value_conflict"


@dataclass(slots=True)
class MemoryCandidate:
    content: str
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    event_time: datetime | None = None
    confidence: float = 1.0
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = "default"
    session_id: str = "default"
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = ""
    embedding: list[float] | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    event_time: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None
    importance: float = 0.5
    utility: float = 0.5
    confidence: float = 1.0
    access_count: int = 0
    last_access_time: datetime | None = None
    version_group: str = field(default_factory=lambda: str(uuid4()))
    version: int = 1
    status: MemoryStatus = MemoryStatus.ACTIVE
    source_ids: list[str] = field(default_factory=list)
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["status"] = self.status.value
        for key in ("event_time", "created_at", "valid_from", "valid_to", "last_access_time"):
            value = data[key]
            data[key] = value.isoformat() if value is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        payload = dict(data)
        payload["memory_type"] = MemoryType(payload.get("memory_type", MemoryType.EPISODIC.value))
        payload["status"] = MemoryStatus(payload.get("status", MemoryStatus.ACTIVE.value))
        for key in ("event_time", "created_at", "valid_from", "valid_to", "last_access_time"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                payload[key] = datetime.fromisoformat(value)
        return cls(**payload)


@dataclass(slots=True)
class SearchResult:
    memory_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WriteResult:
    action: MemoryAction
    memory_id: str | None
    reason: str
    version: int | None = None
    replaced_memory_id: str | None = None


@dataclass(slots=True)
class ReadResult:
    query: str
    hits: list[SearchResult]
    context: str
    query_time: datetime | None = None
