from __future__ import annotations

from collections import defaultdict

from memory.schema import SearchResult


class HybridRetriever:
    """V1 Vector + BM25 retrieval with Reciprocal Rank Fusion."""

    def __init__(self, vector_retriever, bm25_retriever, rrf_k: int = 60, candidate_multiplier: int = 4):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[SearchResult]:
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        vector_hits = self.vector_retriever.search(query, candidate_k, user_id=user_id)
        bm25_hits = self.bm25_retriever.search(query, candidate_k, user_id=user_id)
        scores: dict[str, float] = defaultdict(float)
        content: dict[str, str] = {}
        detail: dict[str, dict] = defaultdict(dict)
        for name, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.memory_id] += 1.0 / (self.rrf_k + rank)
                content[hit.memory_id] = hit.content
                detail[hit.memory_id][f"{name}_rank"] = rank
                detail[hit.memory_id][f"{name}_score"] = hit.score
        ordered = sorted(scores, key=lambda mid: (-scores[mid], mid))[:top_k]
        return [
            SearchResult(
                memory_id=mid,
                content=content[mid],
                score=scores[mid],
                metadata={"retriever": "hybrid-rrf-v1", **detail[mid]},
            )
            for mid in ordered
        ]
