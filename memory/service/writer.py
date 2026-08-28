from __future__ import annotations

from memory.classification import RuleMemoryClassifier
from memory.extraction import RuleMemoryExtractor
from memory.governance import ConflictDetector, DedupDecision, Deduplicator, VersionedMemoryUpdater
from memory.schema import MemoryAction, MemoryRecord, WriteResult
from memory.scoring import ImportanceScorer
from memory.storage import MemoryStore


class MemoryWriterV1:
    """Executable V1 write pipeline.

    Extraction -> Classification -> Importance -> Dedup -> Conflict -> Add/Version Update.
    """

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

    def write(self, messages, user_id: str = "default", session_id: str = "default") -> list[WriteResult]:
        outputs: list[WriteResult] = []
        for candidate in self.extractor.extract(messages):
            memory_type, class_conf = self.classifier.classify(candidate)
            record = MemoryRecord(
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                content=candidate.content,
                entities=candidate.entities,
                keywords=candidate.keywords,
                event_time=candidate.event_time,
                importance=self.importance_scorer.score(candidate),
                confidence=min(candidate.confidence, class_conf),
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object_value,
                metadata={**candidate.metadata, "classifier_confidence": class_conf},
            )
            active = self.store.list_active(user_id=user_id)
            dedup = self.deduplicator.check(record, active)
            if dedup.decision in {DedupDecision.EXACT_DUPLICATE, DedupDecision.SEMANTIC_DUPLICATE}:
                outputs.append(
                    WriteResult(MemoryAction.IGNORE, dedup.matched_memory_id, f"deduplicated: {dedup.decision.value}")
                )
                continue
            conflict = self.conflict_detector.detect(record, active)
            if conflict.is_conflict and conflict.old_memory_id:
                old = self.store.get(conflict.old_memory_id)
                if old is None:
                    raise RuntimeError("conflict target disappeared")
                new = self.updater.supersede(old, record)
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
