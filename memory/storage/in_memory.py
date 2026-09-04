from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from memory.schema import MemoryRecord, MemoryStatus, coerce_datetime
from .base import MemoryStore


class InMemoryMemoryStore(MemoryStore):
    def __init__(self):
        self._records: dict[str, MemoryRecord] = {}
        self._fact_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self._indexed_fact_key: dict[str, tuple[str, str, str]] = {}

    @staticmethod
    def _fact_key(record: MemoryRecord) -> tuple[str, str, str] | None:
        if not record.subject or not record.predicate:
            return None
        return (record.user_id, record.subject, record.predicate)

    def _index(self, record: MemoryRecord) -> None:
        key = self._fact_key(record)
        if key is None:
            self._indexed_fact_key.pop(record.memory_id, None)
            return
        self._fact_ids[key].add(record.memory_id)
        self._indexed_fact_key[record.memory_id] = key

    def _unindex(self, memory_id: str) -> None:
        key = self._indexed_fact_key.pop(memory_id, None)
        if key is None:
            return
        ids = self._fact_ids.get(key)
        if ids is None:
            return
        ids.discard(memory_id)
        if not ids:
            self._fact_ids.pop(key, None)

    def add(self, record: MemoryRecord) -> str:
        if record.memory_id in self._records:
            raise ValueError(f"memory_id already exists: {record.memory_id}")
        self._records[record.memory_id] = record
        self._index(record)
        return record.memory_id

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def update(self, record: MemoryRecord) -> None:
        if record.memory_id not in self._records:
            raise KeyError(record.memory_id)
        self._unindex(record.memory_id)
        self._records[record.memory_id] = record
        self._index(record)

    def list_all(self, user_id: str | None = None) -> list[MemoryRecord]:
        records = list(self._records.values())
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        return sorted(records, key=lambda r: (r.created_at, r.memory_id))

    @staticmethod
    def _visible(record: MemoryRecord, query_time: datetime | str | None) -> bool:
        if query_time is None:
            return record.status == MemoryStatus.ACTIVE
        if record.status == MemoryStatus.ARCHIVED:
            return False
        point = coerce_datetime(query_time)
        valid_from = coerce_datetime(record.valid_from)
        valid_to = coerce_datetime(record.valid_to)
        return bool(
            point is not None
            and valid_from is not None
            and valid_from <= point
            and (valid_to is None or point < valid_to)
        )

    def list_fact_keys(
        self,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[tuple[str, str]]:
        keys = set()
        for (record_user, subject, predicate), memory_ids in self._fact_ids.items():
            if user_id is not None and record_user != user_id:
                continue
            if any(
                self._visible(self._records[memory_id], query_time)
                for memory_id in memory_ids
            ):
                keys.add((subject, predicate))
        return sorted(keys)

    def list_by_fact_key(
        self,
        subject: str,
        predicate: str,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[MemoryRecord]:
        keys = (
            [(user_id, subject, predicate)]
            if user_id is not None
            else [
                key
                for key in self._fact_ids
                if key[1] == subject and key[2] == predicate
            ]
        )
        records = [
            self._records[memory_id]
            for key in keys
            for memory_id in self._fact_ids.get(key, ())
            if self._visible(self._records[memory_id], query_time)
        ]
        return sorted(records, key=lambda record: (record.valid_from, record.memory_id))
