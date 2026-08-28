from __future__ import annotations

from memory.schema import MemoryRecord
from .base import MemoryStore


class InMemoryMemoryStore(MemoryStore):
    def __init__(self):
        self._records: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> str:
        if record.memory_id in self._records:
            raise ValueError(f"memory_id already exists: {record.memory_id}")
        self._records[record.memory_id] = record
        return record.memory_id

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def update(self, record: MemoryRecord) -> None:
        if record.memory_id not in self._records:
            raise KeyError(record.memory_id)
        self._records[record.memory_id] = record

    def list_all(self, user_id: str | None = None) -> list[MemoryRecord]:
        records = list(self._records.values())
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        return sorted(records, key=lambda r: (r.created_at, r.memory_id))
