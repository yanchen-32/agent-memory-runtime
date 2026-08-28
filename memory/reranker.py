from __future__ import annotations

from datetime import datetime

from memory.schema import SearchResult
from memory.scoring import RecencyScorer
from memory.storage import MemoryStore


class WeightedReranker:
    """Explainable feature reranker for V1.

    Version validity remains a hard filter because the store only exposes
    active memories to the retrievers.
    """

    def __init__(
        self,
        store: MemoryStore,
        recency_scorer: RecencyScorer | None = None,
        retrieval_weight: float = 0.55,
        importance_weight: float = 0.15,
        utility_weight: float = 0.10,
        recency_weight: float = 0.10,
        entity_weight: float = 0.10,
    ):
        self.store = store
        self.recency_scorer = recency_scorer or RecencyScorer()
        self.weights = {
            "retrieval": retrieval_weight,
            "importance": importance_weight,
            "utility": utility_weight,
            "recency": recency_weight,
            "entity": entity_weight,
        }

    @staticmethod
    def _normalize_scores(results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        values = [r.score for r in results]
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return {r.memory_id: 1.0 for r in results}
        return {r.memory_id: (r.score - lo) / (hi - lo) for r in results}

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5, now: datetime | None = None) -> list[SearchResult]:
        normalized = self._normalize_scores(results)
        reranked: list[SearchResult] = []
        q = query.lower()
        for hit in results:
            record = self.store.get(hit.memory_id)
            if record is None:
                continue
            entity_match = 1.0 if record.entities and any(e.lower() in q for e in record.entities) else 0.0
            recency = self.recency_scorer.score(record.last_access_time or record.created_at, now=now)
            features = {
                "retrieval": normalized[hit.memory_id],
                "importance": record.importance,
                "utility": record.utility,
                "recency": recency,
                "entity": entity_match,
            }
            final_score = sum(self.weights[k] * features[k] for k in self.weights)
            reranked.append(
                SearchResult(
                    memory_id=hit.memory_id,
                    content=hit.content,
                    score=final_score,
                    metadata={**hit.metadata, "reranker": "weighted-v1", "features": features},
                )
            )
        return sorted(reranked, key=lambda r: (-r.score, r.memory_id))[:top_k]
