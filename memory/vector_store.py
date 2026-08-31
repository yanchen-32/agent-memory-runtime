from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np

from .embedding import EmbeddingModel


@dataclass(slots=True)
class VectorMemoryRecord:
    memory_id: str
    content: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class MemoryHit:
    memory_id: str
    content: str
    score: float
    metadata: dict


class VectorMemoryStore:
    """In-memory cosine vector store for B2.

    Deliberately minimal. No BM25, no validity filter, no versions, no lifecycle.
    """

    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder
        self._records: list[VectorMemoryRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, content: str, metadata: dict | None = None, memory_id: str | None = None) -> str:
        vector = self.embedder.encode([content])[0]
        record = VectorMemoryRecord(
            memory_id=memory_id or str(uuid4()),
            content=content,
            embedding=vector,
            metadata=metadata or {},
        )
        self._records.append(record)
        return record.memory_id

    def add_many(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
        memory_ids: list[str] | None = None,
    ) -> list[str]:
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")
        if memory_ids is not None and len(memory_ids) != len(texts):
            raise ValueError("memory_ids length must match texts length")
        vectors = self.embedder.encode(texts)
        ids: list[str] = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            memory_id = memory_ids[i] if memory_ids is not None else str(uuid4())
            self._records.append(
                VectorMemoryRecord(
                    memory_id=memory_id,
                    content=text,
                    embedding=vector,
                    metadata=(metadata[i] if metadata is not None else {}),
                )
            )
            ids.append(memory_id)
        return ids

    def search(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        if top_k <= 0 or not self._records:
            return []
        query_vec = self.embedder.encode([query])[0]
        matrix = np.stack([r.embedding for r in self._records])
        scores = matrix @ query_vec
        k = min(top_k, len(self._records))
        indices = np.argsort(-scores)[:k]
        return [
            MemoryHit(
                memory_id=self._records[i].memory_id,
                content=self._records[i].content,
                score=float(scores[i]),
                metadata=self._records[i].metadata,
            )
            for i in indices
        ]
