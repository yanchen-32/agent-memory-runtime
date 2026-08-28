from __future__ import annotations

from dataclasses import dataclass

from memory.schema import ConflictType, MemoryRecord


@dataclass(slots=True)
class ConflictResult:
    conflict_type: ConflictType
    old_memory_id: str | None = None
    reason: str = ""

    @property
    def is_conflict(self) -> bool:
        return self.conflict_type != ConflictType.NONE


class ConflictDetector:
    """Deterministic SPO conflict detector for V1."""

    def detect(self, new: MemoryRecord, existing: list[MemoryRecord]) -> ConflictResult:
        if not (new.subject and new.predicate and new.object_value is not None):
            return ConflictResult(ConflictType.NONE, reason="new memory has no structured fact")
        for old in existing:
            if (
                old.subject == new.subject
                and old.predicate == new.predicate
                and old.object_value is not None
                and old.object_value != new.object_value
            ):
                return ConflictResult(
                    ConflictType.VALUE_CONFLICT,
                    old_memory_id=old.memory_id,
                    reason=f"same subject/predicate, value changed: {old.object_value!r} -> {new.object_value!r}",
                )
        return ConflictResult(ConflictType.NONE, reason="no incompatible active fact")
