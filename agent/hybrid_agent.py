from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from .base import Agent, LLMClient, estimate_tokens, select_context_indices
from memory.embedding import EmbeddingModel, HashEmbeddingModel
from memory.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from memory.schema import MemoryRecord, coerce_datetime, utcnow
from memory.storage import InMemoryMemoryStore


class HybridMemoryAgent(Agent):
    """B3 baseline: vector + BM25 candidates fused by RRF.

    This baseline deliberately excludes Runtime V1 governance, reranking,
    conflict resolution and consolidation so that B3 remains comparable as an
    independent retrieval baseline.
    """

    def __init__(
        self,
        client: LLMClient,
        embedder: EmbeddingModel | None = None,
        top_k: int = 5,
    ):
        self.client = client
        self.store = InMemoryMemoryStore()
        self.embedder = embedder or HashEmbeddingModel(dim=384)
        self.vector_retriever = VectorRetriever(self.store, self.embedder)
        self.bm25_retriever = BM25Retriever(self.store)
        self.retriever = HybridRetriever(self.vector_retriever, self.bm25_retriever)
        self.top_k = top_k
        self.last_prompt = ""
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []
        self.last_prompt_tokens = 0

    def ingest(self, messages: list[dict]) -> list[str]:
        memory_ids: list[str] = []
        for index, message in enumerate(messages):
            content = str(message.get("content", ""))
            if not content:
                continue
            memory_id = str(message.get("memory_id") or f"b3-{index}-{uuid4().hex[:8]}")
            created_at = coerce_datetime(
                message.get("created_at") or message.get("valid_from")
            )
            timestamp = created_at or utcnow()
            record = MemoryRecord(
                memory_id=memory_id,
                user_id="benchmark",
                session_id="b3",
                content=content,
                created_at=timestamp,
                valid_from=timestamp,
                metadata={"role": message.get("role", "user"), "baseline": "B3"},
            )
            self.store.add(record)
            memory_ids.append(memory_id)
        return memory_ids

    def answer(
        self,
        query: str,
        token_budget: int | None = None,
        query_time: datetime | str | None = None,
    ) -> str:
        # query_time is intentionally ignored: B3 is a retrieval-only
        # baseline and does not claim temporal/version governance.
        del query_time
        hits = self.retriever.search(query, top_k=self.top_k)
        lines = [f"MEMORY[{i}] {hit.content}" for i, hit in enumerate(hits, start=1)]
        header = (
            "You are a hybrid-memory baseline agent.\n"
            "Use only the retrieved memory below. If the memory does not support an answer, answer UNKNOWN.\n\n"
        )
        indices = select_context_indices(lines, query, token_budget, header)
        selected_hits = [hits[i] for i in indices]
        self.last_context = "\n".join(lines[i] for i in indices)
        self.last_retrieved_ids = [hit.memory_id for hit in selected_hits]
        self.last_prompt = header + self.last_context + "\n\n" + f"QUESTION: {query}\n"
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        return self.client.generate(self.last_prompt)
