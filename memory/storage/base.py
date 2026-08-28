from __future__ import annotations

from abc import ABC, abstractmethod

from memory.schema import MemoryRecord, MemoryStatus


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
