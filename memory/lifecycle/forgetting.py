from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from memory.schema import MemoryRecord, MemoryStatus
from memory.scoring import RecencyScorer
from memory.storage import MemoryStore


@dataclass(slots=True)
class StrengthBreakdown:
    score: float
    importance: float
    utility: float
    recency: float
    access: float


class ForgettingPolicy:
    """V1 lifecycle policy: low-strength active memories are archived, not deleted."""

    def __init__(self, store: MemoryStore, threshold: float = 0.25, recency_scorer: RecencyScorer | None = None):
        self.store = store
        self.threshold = threshold
        self.recency_scorer = recency_scorer or RecencyScorer(decay_rate_per_day=0.02)

    def strength(self, record: MemoryRecord, now: datetime | None = None) -> StrengthBreakdown:
        recency = self.recency_scorer.score(record.last_access_time or record.created_at, now=now)
        access = min(1.0, math.log1p(max(0, record.access_count)) / math.log(11.0))
        score = 0.40 * record.importance + 0.25 * record.utility + 0.20 * recency + 0.15 * access
        return StrengthBreakdown(score, record.importance, record.utility, recency, access)

    def evaluate(self, record: MemoryRecord, now: datetime | None = None) -> bool:
        return record.status == MemoryStatus.ACTIVE and self.strength(record, now=now).score < self.threshold

    def archive(self, record: MemoryRecord) -> None:
        record.status = MemoryStatus.ARCHIVED
        self.store.update(record)

    def run(self, user_id: str | None = None, now: datetime | None = None) -> list[str]:
        archived: list[str] = []
        for record in self.store.list_active(user_id=user_id):
            if self.evaluate(record, now=now):
                self.archive(record)
                archived.append(record.memory_id)
        return archived
