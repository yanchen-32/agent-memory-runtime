from __future__ import annotations

from memory.embedding import EmbeddingModel, HashEmbeddingModel
from memory.governance import Deduplicator
from memory.lifecycle import ForgettingPolicy, MemoryCompressor
from memory.reranker import WeightedReranker
from memory.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from memory.service import MemoryReaderV1, MemoryWriterV1
from memory.storage import InMemoryMemoryStore, MemoryStore


class MemoryRuntimeV1:
    """Convenience facade wiring all V1 modules with dependency injection."""

    def __init__(self, store: MemoryStore | None = None, embedder: EmbeddingModel | None = None):
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

    def write(self, messages, user_id: str = "default", session_id: str = "default"):
        return self.writer.write(messages, user_id=user_id, session_id=session_id)

    def read(self, query: str, top_k: int = 5, user_id: str | None = None):
        return self.reader.read(query, top_k=top_k, user_id=user_id)
