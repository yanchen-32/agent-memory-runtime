from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

from memory.embedding import EmbeddingModel
from memory.schema import MemoryRecord


class DedupDecision(str, Enum):
    UNIQUE = "unique"
    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    POSSIBLE_CONFLICT = "possible_conflict"


@dataclass(slots=True)
class DedupResult:
    decision: DedupDecision
    matched_memory_id: str | None = None
    similarity: float = 0.0


class Deduplicator:
    def __init__(self, embedder: EmbeddingModel, similarity_threshold: float = 0.90):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+|[，。！？,.!?:：;；]", "", text).lower()

    def check(self, new: MemoryRecord, existing: list[MemoryRecord]) -> DedupResult:
        for old in existing:
            if self.normalize(new.content) == self.normalize(old.content):
                return DedupResult(DedupDecision.EXACT_DUPLICATE, old.memory_id, 1.0)
        if not existing:
            return DedupResult(DedupDecision.UNIQUE)
        vectors = self.embedder.encode([new.content] + [r.content for r in existing])
        new_vec, old_vecs = vectors[0], vectors[1:]
        similarities = old_vecs @ new_vec
        idx = int(np.argmax(similarities))
        similarity = float(similarities[idx])
        old = existing[idx]
        if similarity < self.similarity_threshold:
            return DedupResult(DedupDecision.UNIQUE, old.memory_id, similarity)
        same_fact_key = bool(new.subject and new.predicate and new.subject == old.subject and new.predicate == old.predicate)
        if same_fact_key and new.object_value != old.object_value:
            return DedupResult(DedupDecision.POSSIBLE_CONFLICT, old.memory_id, similarity)
        return DedupResult(DedupDecision.SEMANTIC_DUPLICATE, old.memory_id, similarity)
