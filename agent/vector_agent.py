from __future__ import annotations

import time

from .base import (
    ANSWER_FORMAT_INSTRUCTION,
    Agent,
    LLMClient,
    estimate_tokens,
    memory_line,
    select_context_indices,
    temporal_header,
    timed_generate,
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

    def answer(
        self,
        query: str,
        token_budget: int | None = None,
        query_time: str | None = None,
        temporal_context: bool = False,
    ) -> str:
        memory_started = time.perf_counter()
        hits = self.store.search(query, top_k=self.top_k)
        self.last_memory_latency_ms = (time.perf_counter() - memory_started) * 1000
        context_started = time.perf_counter()
        lines = [
            memory_line(
                i,
                hit.content,
                temporal_context=temporal_context,
                valid_from=hit.metadata.get("valid_from"),
                valid_to=hit.metadata.get("valid_to"),
            )
            for i, hit in enumerate(hits, start=1)
        ]
        header = (
            "You are a vector-memory baseline agent.\n"
            "Use only the retrieved memory below.\n"
            + temporal_header(query_time, temporal_context)
            + ANSWER_FORMAT_INSTRUCTION
            + "\n"
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
        self.last_context_latency_ms = (time.perf_counter() - context_started) * 1000
        response, self.last_llm_latency_ms = timed_generate(self.client, self.last_prompt)
        return response
