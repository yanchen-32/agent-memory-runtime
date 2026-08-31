from __future__ import annotations

from datetime import datetime

from memory.classification import RuleMemoryClassifier
from memory.extraction import RuleMemoryExtractor
from memory.governance import ConflictDetector, DedupDecision, Deduplicator, VersionedMemoryUpdater
from memory.schema import (
    MemoryAction,
    MemoryRecord,
    MemoryType,
    WriteResult,
    coerce_datetime,
    utcnow,
)
from memory.scoring import ImportanceScorer
from memory.storage import MemoryStore


class MemoryWriterV1:
    """Executable V1 write pipeline with controllable validity timestamps."""

    def __init__(
        self,
        store: MemoryStore,
        deduplicator: Deduplicator,
        extractor=None,
        classifier=None,
        importance_scorer=None,
        conflict_detector=None,
        updater=None,
    ):
        self.store = store
        self.extractor = extractor or RuleMemoryExtractor()
        self.classifier = classifier or RuleMemoryClassifier()
        self.importance_scorer = importance_scorer or ImportanceScorer()
        self.deduplicator = deduplicator
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.updater = updater or VersionedMemoryUpdater(store)

    def write(
        self,
        messages,
        user_id: str = "default",
        session_id: str = "default",
        at: datetime | str | None = None,
        memory_type_override: MemoryType | None = None,
        preserve_duplicates: bool = False,
    ) -> list[WriteResult]:
        outputs: list[WriteResult] = []
        write_time = coerce_datetime(at) or utcnow()
        for candidate in self.extractor.extract(messages):
            memory_type, class_conf = self.classifier.classify(candidate)
            if memory_type_override is not None:
                memory_type = memory_type_override
            record = MemoryRecord(
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                content=candidate.content,
                entities=candidate.entities,
                keywords=candidate.keywords,
                event_time=candidate.event_time,
                created_at=write_time,
                valid_from=write_time,
                importance=self.importance_scorer.score(candidate),
                confidence=min(candidate.confidence, class_conf),
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object_value,
                metadata={**candidate.metadata, "classifier_confidence": class_conf},
            )
            active = self.store.list_active(user_id=user_id)
            dedup = self.deduplicator.check(record, active)
            if (
                not preserve_duplicates
                and dedup.decision in {
                    DedupDecision.EXACT_DUPLICATE,
                    DedupDecision.SEMANTIC_DUPLICATE,
                }
            ):
                outputs.append(
                    WriteResult(
                        MemoryAction.IGNORE,
                        dedup.matched_memory_id,
                        f"deduplicated: {dedup.decision.value}",
                    )
                )
                continue
            conflict = self.conflict_detector.detect(record, active)
            if conflict.is_conflict and conflict.old_memory_id:
                old = self.store.get(conflict.old_memory_id)
                if old is None:
                    raise RuntimeError("conflict target disappeared")
                new = self.updater.supersede(old, record, at=write_time)
                outputs.append(
                    WriteResult(
                        MemoryAction.SUPERSEDE,
                        new.memory_id,
                        conflict.reason,
                        version=new.version,
                        replaced_memory_id=old.memory_id,
                    )
                )
                continue
            self.store.add(record)
            outputs.append(WriteResult(MemoryAction.ADD, record.memory_id, "new memory", version=record.version))
        return outputs
