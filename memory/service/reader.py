from __future__ import annotations

from datetime import datetime, timezone

from memory.lifecycle import MemoryCompressor
from memory.schema import ReadResult, coerce_datetime
from memory.storage import MemoryStore


class MemoryReaderV1:
    """Executable V1 read pipeline with current and historical query modes."""

    def __init__(self, store: MemoryStore, retriever, reranker, compressor: MemoryCompressor | None = None):
        self.store = store
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor or MemoryCompressor()
        self.last_candidates = []
        self.last_hits = []

    def read(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> ReadResult:
        point = coerce_datetime(query_time)
        candidates = self.retriever.search(
            query,
            top_k=max(20, top_k * 4),
            user_id=user_id,
            query_time=point,
        )
        self.last_candidates = list(candidates)
        hits = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
            now=point or datetime.now(timezone.utc),
        )
        self.last_hits = list(hits)
        context_parts: list[str] = []
        now = datetime.now(timezone.utc)
        for hit in hits:
            record = self.store.get(hit.memory_id)
            if record is None:
                continue
            record.access_count += 1
            record.last_access_time = now
            self.store.update(record)
            compressed = self.compressor.compress(record)
            hit.metadata["compressed"] = compressed.compressed
            hit.metadata["compression_ratio"] = compressed.ratio
            context_parts.append(f"[{record.memory_id}] {compressed.text}")
        return ReadResult(
            query=query,
            hits=hits,
            context="\n".join(context_parts),
            query_time=point,
        )
