from __future__ import annotations

from datetime import datetime

from memory.context_budget import BudgetSelection, ContextBudgetManager
from memory.consolidation import ConsolidationReport, MemoryConsolidator
from memory.embedding import EmbeddingModel, HashEmbeddingModel
from memory.governance import Deduplicator
from memory.lifecycle import ForgettingPolicy, MemoryCompressor
from memory.reranker import WeightedReranker
from memory.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from memory.schema import ReadResult, coerce_datetime
from memory.service import MemoryReaderV1, MemoryWriterV1
from memory.storage import InMemoryMemoryStore, MemoryStore


class MemoryRuntimeV1:
    """Convenience facade wiring memory, temporal and budget components."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedder: EmbeddingModel | None = None,
        budget_manager: ContextBudgetManager | None = None,
    ):
        self.store = store or InMemoryMemoryStore()
        self.embedder = embedder or HashEmbeddingModel(dim=384)

        self.vector_retriever = VectorRetriever(self.store, self.embedder)
        self.bm25_retriever = BM25Retriever(self.store)
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.bm25_retriever)
        self.reranker = WeightedReranker(self.store)
        self.compressor = MemoryCompressor()
        self.deduplicator = Deduplicator(self.embedder)

        self.writer = MemoryWriterV1(self.store, self.deduplicator)
        self.reader = MemoryReaderV1(self.store, self.hybrid_retriever, self.reranker, self.compressor)
        self.forgetting = ForgettingPolicy(self.store)
        self.budget_manager = budget_manager or ContextBudgetManager(self.store)
        self.consolidator = MemoryConsolidator(self.store)

    def write(
        self,
        messages,
        user_id: str = "default",
        session_id: str = "default",
        at: datetime | str | None = None,
    ):
        return self.writer.write(messages, user_id=user_id, session_id=session_id, at=at)

    def read(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> ReadResult:
        return self.reader.read(
            query,
            top_k=top_k,
            user_id=user_id,
            query_time=coerce_datetime(query_time),
        )

    def select_context(
        self,
        query: str,
        result: ReadResult,
        token_budget: int | None,
        prefix: str = "",
        suffix: str = "",
    ) -> BudgetSelection:
        lines = [
            f"MEMORY[{i}] {hit.content}"
            for i, hit in enumerate(result.hits, start=1)
        ]
        return self.budget_manager.select(
            query=query,
            candidates=result.hits,
            context_lines=lines,
            token_budget=token_budget,
            prefix=prefix,
            suffix=suffix,
        )

    def consolidate(
        self,
        user_id: str | None = None,
    ) -> ConsolidationReport:
        """Consolidate active episodic memories and keep source traceability."""
        return self.consolidator.consolidate(user_id=user_id)
