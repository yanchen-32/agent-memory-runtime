from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from memory.schema import MemoryRecord, MemoryStatus, coerce_datetime


class MemoryStore(ABC):
    @abstractmethod
    def add(self, record: MemoryRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, memory_id: str) -> MemoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    def update(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_all(self, user_id: str | None = None) -> list[MemoryRecord]:
        raise NotImplementedError

    def list_active(self, user_id: str | None = None) -> list[MemoryRecord]:
        return [r for r in self.list_all(user_id=user_id) if r.status == MemoryStatus.ACTIVE]

    def list_valid_at(
        self,
        query_time: datetime | str,
        user_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Return versions valid at a point in time.

        Superseded versions remain queryable historically. Archived records are
        excluded because lifecycle archival removes them from the usable memory
        surface. The interval is [valid_from, valid_to).
        """
        point = coerce_datetime(query_time)
        if point is None:
            raise ValueError("query_time is required")
        records: list[MemoryRecord] = []
        for record in self.list_all(user_id=user_id):
            if record.status == MemoryStatus.ARCHIVED:
                continue
            valid_from = coerce_datetime(record.valid_from)
            valid_to = coerce_datetime(record.valid_to)
            if valid_from is not None and valid_from <= point:
                if valid_to is None or point < valid_to:
                    records.append(record)
        return records

    def list_fact_keys(
        self,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[tuple[str, str]]:
        """List unique structured fact keys visible at ``query_time``.

        Stores may override this with an index.  The default keeps the storage
        interface backwards compatible for persistent adapters.
        """
        records = (
            self.list_valid_at(query_time, user_id=user_id)
            if query_time is not None
            else self.list_active(user_id=user_id)
        )
        return sorted({
            (record.subject, record.predicate)
            for record in records
            if record.subject and record.predicate
        })

    def list_by_fact_key(
        self,
        subject: str,
        predicate: str,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[MemoryRecord]:
        """Read visible records for an exact ``subject + predicate`` key."""
        records = (
            self.list_valid_at(query_time, user_id=user_id)
            if query_time is not None
            else self.list_active(user_id=user_id)
        )
        return [
            record
            for record in records
            if record.subject == subject and record.predicate == predicate
        ]
