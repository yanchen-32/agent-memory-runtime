from __future__ import annotations

from .base import Agent, LLMClient
from memory.vector_store import VectorMemoryStore


class VectorMemoryAgent(Agent):
    """B2 baseline: embedding -> vector search -> Top-K prompt injection."""

    def __init__(self, client: LLMClient, store: VectorMemoryStore, top_k: int = 5):
        self.client = client
        self.store = store
        self.top_k = top_k

    def ingest(self, texts: list[str], metadata: list[dict] | None = None) -> list[str]:
        return self.store.add_many(texts, metadata=metadata)

    def answer(self, query: str) -> str:
        hits = self.store.search(query, top_k=self.top_k)
        memory_block = "\n".join(
            f"MEMORY[{i}] {hit.content}" for i, hit in enumerate(hits, start=1)
        )
        prompt = (
            "You are a vector-memory baseline agent.\n"
            "Use only the retrieved memory below. If the memory does not support an answer, answer UNKNOWN.\n\n"
            f"{memory_block}\n\n"
            f"QUESTION: {query}\n"
        )
        return self.client.generate(prompt)
