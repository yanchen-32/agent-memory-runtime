from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re

from memory.context_budget import estimate_tokens
from memory.schema import MemoryRecord


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))


def _similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


@dataclass(slots=True)
class ConsolidationDecision:
    should_consolidate: bool
    trigger_score: float
    granularity_score: float
    granularity_level: str
    conflict_risk: float
    reason: str
    features: dict[str, float] = field(default_factory=dict)


class AdaptiveConsolidationPolicy:
    """Explainable V1 trigger and granularity policy.

    Defaults are development parameters, not learned values. Formal use must
    record this policy version and freeze all weights before Test/Holdout.
    """

    version = "adaptive-rule-v1"

    def __init__(
        self,
        trigger_threshold: float = 0.35,
        conflict_threshold: float = 0.5,
        target_cluster_size: int = 4,
        age_horizon_days: float = 30.0,
        storage_token_capacity: int = 100_000,
        trigger_weights: dict[str, float] | None = None,
        granularity_weights: dict[str, float] | None = None,
        fine_threshold: float = 0.67,
        coarse_threshold: float = 0.34,
    ):
        self.trigger_threshold = trigger_threshold
        self.conflict_threshold = conflict_threshold
        self.target_cluster_size = max(1, target_cluster_size)
        self.age_horizon_days = max(1e-9, age_horizon_days)
        self.storage_token_capacity = max(1, storage_token_capacity)
        self.trigger_weights = trigger_weights or {
            "redundancy": 0.35,
            "cluster_size": 0.30,
            "age": 0.20,
            "storage_pressure": 0.15,
        }
        self.granularity_weights = granularity_weights or {
            "importance": 0.30,
            "novelty": 0.25,
            "information_density": 0.20,
            "access_frequency": 0.15,
            "redundancy": 0.10,
        }
        self.fine_threshold = fine_threshold
        self.coarse_threshold = coarse_threshold

    @staticmethod
    def _mean_pairwise_similarity(records: list[MemoryRecord]) -> float:
        values = [
            _similarity(records[left].content, records[right].content)
            for left in range(len(records))
            for right in range(left + 1, len(records))
        ]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _information_density(records: list[MemoryRecord]) -> float:
        facts = sum(
            1 if record.subject and record.predicate and record.object_value else
            max(1, len(re.findall(r"[；;。.!！？?]+", record.content)))
            for record in records
        )
        tokens = sum(max(1, estimate_tokens(record.content)) for record in records)
        return min(1.0, 10.0 * facts / tokens)

    def decide(
        self,
        records: list[MemoryRecord],
        semantic_memories: list[MemoryRecord] | None = None,
        storage_pressure: float | None = None,
        conflict_risk: float = 0.0,
        now: datetime | None = None,
    ) -> ConsolidationDecision:
        if not records:
            raise ValueError("records must not be empty")
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        redundancy = self._mean_pairwise_similarity(records)
        cluster_size = min(1.0, len(records) / self.target_cluster_size)
        oldest = min(record.created_at for record in records)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - oldest).total_seconds() / 86_400.0)
        age = min(1.0, age_days / self.age_horizon_days)
        if storage_pressure is None:
            token_count = sum(estimate_tokens(record.content) for record in records)
            storage_pressure = min(1.0, token_count / self.storage_token_capacity)
        storage_pressure = min(1.0, max(0.0, storage_pressure))

        trigger_features = {
            "redundancy": redundancy,
            "cluster_size": cluster_size,
            "age": age,
            "storage_pressure": storage_pressure,
        }
        trigger_score = sum(
            self.trigger_weights[name] * value
            for name, value in trigger_features.items()
        )

        importance = sum(record.importance for record in records) / len(records)
        existing = semantic_memories or []
        source_text = " ".join(record.content for record in records)
        novelty = 1.0 - max(
            (_similarity(source_text, record.content) for record in existing),
            default=0.0,
        )
        information_density = self._information_density(records)
        access_frequency = sum(
            min(1.0, math.log1p(max(0, record.access_count)) / math.log(11.0))
            for record in records
        ) / len(records)
        granularity_features = {
            "importance": importance,
            "novelty": novelty,
            "information_density": information_density,
            "access_frequency": access_frequency,
            "redundancy": redundancy,
        }
        granularity_score = (
            self.granularity_weights["importance"] * importance
            + self.granularity_weights["novelty"] * novelty
            + self.granularity_weights["information_density"] * information_density
            + self.granularity_weights["access_frequency"] * access_frequency
            - self.granularity_weights["redundancy"] * redundancy
        )
        if granularity_score >= self.fine_threshold:
            level = "fine"
        elif granularity_score < self.coarse_threshold:
            level = "coarse"
        else:
            level = "normal"

        if conflict_risk >= self.conflict_threshold:
            should_consolidate = False
            reason = "conflict_requires_version_governance"
        elif trigger_score < self.trigger_threshold:
            should_consolidate = False
            reason = "trigger_below_threshold"
        else:
            should_consolidate = True
            reason = "trigger_threshold_met"

        return ConsolidationDecision(
            should_consolidate=should_consolidate,
            trigger_score=trigger_score,
            granularity_score=granularity_score,
            granularity_level=level,
            conflict_risk=conflict_risk,
            reason=reason,
            features={**trigger_features, **granularity_features},
        )
