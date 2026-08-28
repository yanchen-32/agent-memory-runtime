from __future__ import annotations

from datetime import datetime, timezone

from memory.schema import MemoryRecord, MemoryStatus
from memory.storage import MemoryStore


class VersionedMemoryUpdater:
    """V1 version-chain update with validity timestamps wired for later temporal QA."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def supersede(self, old: MemoryRecord, new: MemoryRecord, at: datetime | None = None) -> MemoryRecord:
        at = at or datetime.now(timezone.utc)
        old.status = MemoryStatus.SUPERSEDED
        old.valid_to = at
        self.store.update(old)

        new.version_group = old.version_group
        new.version = old.version + 1
        new.valid_from = at
        new.valid_to = None
        new.status = MemoryStatus.ACTIVE
        self.store.add(new)
        return new
