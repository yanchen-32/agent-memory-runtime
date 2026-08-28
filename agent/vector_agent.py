from __future__ import annotations

from .base import (
    Agent,
    LLMClient,
    estimate_tokens,
    select_context_indices,
)
from memory.vector_store import VectorMemoryStore


class VectorMemoryAgent(Agent):
    """B2 baseline: embedding -> vector search -> Top-K prompt injection."""

    def __init__(self, client: LLMClient, store: VectorMemoryStore, top_k: int = 5):
        self.client = client
        self.store = store
        self.top_k = top_k
        self.last_prompt = ""
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []

    def ingest(self, texts: list[str], metadata: list[dict] | None = None) -> list[str]:
        return self.store.add_many(texts, metadata=metadata)

    def answer(self, query: str, token_budget: int | None = None) -> str:
        hits = self.store.search(query, top_k=self.top_k)
        lines = [
            f"MEMORY[{i}] {hit.content}" for i, hit in enumerate(hits, start=1)
        ]
        header = (
            "You are a vector-memory baseline agent.\n"
            "Use only the retrieved memory below. If the memory does not support an answer, answer UNKNOWN.\n\n"
        )
        indices = select_context_indices(lines, query, token_budget, header)
        selected_hits = [hits[i] for i in indices]
        selected_lines = [lines[i] for i in indices]
        self.last_context = "\n".join(selected_lines)
        self.last_retrieved_ids = [hit.memory_id for hit in selected_hits]
        self.last_prompt = (
            header
            + self.last_context
            + "\n\n"
            + f"QUESTION: {query}\n"
        )
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        return self.client.generate(self.last_prompt)
