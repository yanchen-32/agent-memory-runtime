from __future__ import annotations

import math
from datetime import datetime, timezone


class RecencyScorer:
    def __init__(self, decay_rate_per_day: float = 0.01):
        if decay_rate_per_day < 0:
            raise ValueError("decay_rate_per_day must be >= 0")
        self.decay_rate_per_day = decay_rate_per_day

    def score(self, timestamp: datetime, now: datetime | None = None) -> float:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
        return math.exp(-self.decay_rate_per_day * delta_days)
