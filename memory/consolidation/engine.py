from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Iterable
from uuid import uuid4

from memory.schema import MemoryRecord, MemoryStatus, MemoryType
from memory.storage import MemoryStore
from memory.context_budget import estimate_tokens
from .policy import AdaptiveConsolidationPolicy, ConsolidationDecision


@dataclass(slots=True)
class ConsolidationGroup:
    group_key: str
    source_ids: list[str]
    semantic_memory_id: str | None
    fidelity: float
    action: str
    trigger_score: float | None = None
    granularity_score: float | None = None
    granularity_level: str = "normal"
    policy_version: str = "fixed-v1"
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def compression_ratio(self) -> float:
        return (
            1.0 - self.tokens_after / self.tokens_before
            if self.tokens_before > 0
            else 0.0
        )


@dataclass(slots=True)
class ConsolidationReport:
    groups: list[ConsolidationGroup] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    skipped_groups: int = 0
    skipped_by_policy: int = 0
    conflict_blocked_groups: int = 0

    @property
    def source_count(self) -> int:
        return sum(len(group.source_ids) for group in self.groups)

    @property
    def semantic_count(self) -> int:
        return len(self.created_ids) + len(self.updated_ids)

    @property
    def fidelity(self) -> float:
        return (
            sum(group.fidelity for group in self.groups) / len(self.groups)
            if self.groups
            else 0.0
        )


class MemoryConsolidator:
    """Deterministic Episodic -> Semantic V1 consolidation engine.

    Groups are formed from extracted fact triples when available. Records with
    different object values are kept in different groups, so a known conflict
    is not silently merged into one semantic fact. Episodic sources remain
    intact and semantic records retain source_ids for traceability.
    """

    def __init__(
        self,
        store: MemoryStore,
        min_group_size: int = 2,
        policy: AdaptiveConsolidationPolicy | None = None,
    ):
        if min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        self.store = store
        self.min_group_size = min_group_size
        self.policy = policy

    @staticmethod
    def _normalise(value: str | None) -> str:
        return re.sub(r"\s+", "", (value or "").strip().lower())

    def _group_key(self, record: MemoryRecord) -> str:
        if record.subject and record.predicate and record.object_value:
            return "fact:" + ":".join(
                self._normalise(value)
                for value in (record.subject, record.predicate, record.object_value)
            )
        entities = sorted({self._normalise(value) for value in record.entities if value})
        keywords = sorted({self._normalise(value) for value in record.keywords if value})
        if entities or keywords:
            return "features:" + "|".join(entities + keywords)
        tokens = sorted(set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", record.content.lower())))
        return "text:" + "|".join(tokens[:12])

    @staticmethod
    def _canonical_record(records: list[MemoryRecord]) -> MemoryRecord:
        return sorted(
            records,
            key=lambda record: (
                record.valid_from,
                record.created_at,
                record.memory_id,
            ),
        )[-1]

    def _summary(self, records: list[MemoryRecord], granularity: str = "normal") -> str:
        canonical = self._canonical_record(records)
        if granularity == "fine":
            unique = list(dict.fromkeys(record.content for record in records))
            return "语义记忆：" + "；".join(unique)
        if canonical.subject and canonical.predicate and canonical.object_value:
            return f"{canonical.subject}{canonical.predicate}{canonical.object_value}。"
        if granularity == "coarse":
            return canonical.content
        return "综合记忆：" + "；".join(dict.fromkeys(record.content for record in records))

    @staticmethod
    def _unresolved_conflict_keys(records: list[MemoryRecord]) -> set[tuple[str, str]]:
        values: dict[tuple[str, str], set[str]] = {}
        for record in records:
            if (
                record.status == MemoryStatus.ACTIVE
                and record.subject
                and record.predicate
                and record.object_value is not None
            ):
                key = (record.subject, record.predicate)
                values.setdefault(key, set()).add(record.object_value)
        return {key for key, objects in values.items() if len(objects) > 1}

    def consolidate(
        self,
        user_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        now: datetime | None = None,
    ) -> ConsolidationReport:
        # Superseded episodic records are retained as historical evidence for
        # consolidation; archived records are excluded by lifecycle policy.
        records = [
            record
            for record in self.store.list_all(user_id=user_id)
            if record.status != MemoryStatus.ARCHIVED
            and record.memory_type == memory_type
        ]
        semantic_memories = [
            record
            for record in self.store.list_active(user_id=user_id)
            if record.memory_type == MemoryType.SEMANTIC
        ]
        unresolved_conflicts = self._unresolved_conflict_keys(records)
        total_tokens = sum(max(1, estimate_tokens(record.content)) for record in records)
        storage_pressure = (
            min(1.0, total_tokens / self.policy.storage_token_capacity)
            if self.policy is not None
            else 0.0
        )
        grouped: dict[str, list[MemoryRecord]] = {}
        for record in records:
            grouped.setdefault(self._group_key(record), []).append(record)

        report = ConsolidationReport()
        for group_key, source_records in sorted(grouped.items()):
            if len(source_records) < self.min_group_size:
                report.skipped_groups += 1
                continue
            fact_key = (
                (source_records[0].subject, source_records[0].predicate)
                if source_records[0].subject and source_records[0].predicate
                else None
            )
            conflict_risk = 1.0 if fact_key in unresolved_conflicts else 0.0
            decision: ConsolidationDecision | None = None
            if self.policy is not None:
                decision = self.policy.decide(
                    source_records,
                    semantic_memories=semantic_memories,
                    storage_pressure=storage_pressure,
                    conflict_risk=conflict_risk,
                    now=now,
                )
                if not decision.should_consolidate:
                    report.skipped_groups += 1
                    if decision.reason == "conflict_requires_version_governance":
                        report.conflict_blocked_groups += 1
                    else:
                        report.skipped_by_policy += 1
                    continue
            source_ids = sorted(record.memory_id for record in source_records)
            existing = next(
                (
                    record
                    for record in self.store.list_active(user_id=user_id)
                    if record.memory_type == MemoryType.SEMANTIC
                    and record.metadata.get("consolidation_group") == group_key
                ),
                None,
            )
            canonical = self._canonical_record(source_records)
            granularity = decision.granularity_level if decision else "normal"
            summary = self._summary(source_records, granularity=granularity)
            policy_version = self.policy.version if self.policy else "fixed-v1"
            tokens_before = sum(max(1, estimate_tokens(record.content)) for record in source_records)
            tokens_after = max(1, estimate_tokens(summary))
            if existing is None:
                semantic = MemoryRecord(
                    memory_id=f"semantic-{uuid4().hex}",
                    user_id=canonical.user_id,
                    session_id=canonical.session_id,
                    memory_type=MemoryType.SEMANTIC,
                    content=summary,
                    entities=sorted({entity for r in source_records for entity in r.entities}),
                    keywords=sorted({keyword for r in source_records for keyword in r.keywords}),
                    event_time=canonical.event_time,
                    created_at=canonical.created_at,
                    valid_from=min(record.valid_from for record in source_records),
                    valid_to=None,
                    importance=max(record.importance for record in source_records),
                    utility=max(record.utility for record in source_records),
                    confidence=min(record.confidence for record in source_records),
                    source_ids=source_ids,
                    subject=canonical.subject,
                    predicate=canonical.predicate,
                    object_value=canonical.object_value,
                    metadata={
                        "consolidation_engine": "episodic-to-semantic-v1",
                        "consolidation_group": group_key,
                        "source_count": len(source_ids),
                        "granularity_level": granularity,
                        "consolidation_policy_version": policy_version,
                        "trigger_score": decision.trigger_score if decision else None,
                        "granularity_score": decision.granularity_score if decision else None,
                        "tokens_before": tokens_before,
                        "tokens_after": tokens_after,
                        "compression_ratio": 1.0 - tokens_after / tokens_before,
                    },
                )
                self.store.add(semantic)
                semantic_id = semantic.memory_id
                report.created_ids.append(semantic_id)
                action = "created"
            else:
                existing.source_ids = source_ids
                existing.content = summary
                existing.metadata["source_count"] = len(source_ids)
                existing.metadata.update({
                    "granularity_level": granularity,
                    "consolidation_policy_version": policy_version,
                    "trigger_score": decision.trigger_score if decision else None,
                    "granularity_score": decision.granularity_score if decision else None,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "compression_ratio": 1.0 - tokens_after / tokens_before,
                })
                existing.valid_from = min(record.valid_from for record in source_records)
                self.store.update(existing)
                semantic_id = existing.memory_id
                report.updated_ids.append(semantic_id)
                action = "updated"
            report.groups.append(
                ConsolidationGroup(
                    group_key=group_key,
                    source_ids=source_ids,
                    semantic_memory_id=semantic_id,
                    fidelity=1.0,
                    action=action,
                    trigger_score=decision.trigger_score if decision else None,
                    granularity_score=decision.granularity_score if decision else None,
                    granularity_level=granularity,
                    policy_version=policy_version,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                )
            )
        return report
