from __future__ import annotations

import numpy as np

from memory.embedding import EmbeddingModel
from memory.schema import SearchResult
from memory.storage import MemoryStore


class VectorRetriever:
    """V1 semantic retriever over the full MemoryRecord store."""

    def __init__(self, store: MemoryStore, embedder: EmbeddingModel):
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[SearchResult]:
        records = self.store.list_active(user_id=user_id)
        if top_k <= 0 or not records:
            return []
        q = self.embedder.encode([query])[0]
        missing = [r for r in records if r.embedding is None]
        if missing:
            vectors = self.embedder.encode([r.content for r in missing])
            for record, vector in zip(missing, vectors):
                record.embedding = vector.astype(float).tolist()
                self.store.update(record)
        matrix = np.asarray([r.embedding for r in records], dtype=np.float32)
        scores = matrix @ q
        order = np.argsort(-scores)[: min(top_k, len(records))]
        return [
            SearchResult(
                memory_id=records[i].memory_id,
                content=records[i].content,
                score=float(scores[i]),
                metadata={"retriever": "vector-v1", "memory_type": records[i].memory_type.value},
            )
            for i in order
        ]
