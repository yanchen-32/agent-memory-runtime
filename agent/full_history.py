from __future__ import annotations

import time

from .base import (
    ANSWER_FORMAT_INSTRUCTION,
    Agent,
    LLMClient,
    estimate_tokens,
    select_context_indices,
    timed_generate,
)


class FullHistoryAgent(Agent):
    """B1 baseline: send the complete conversation history to the model.

    B1 deliberately performs no extraction, storage, retrieval, reranking or
    lifecycle management. An optional token budget is exposed only for the
    budget benchmark and is reported as a constrained full-history run.
    """

    def __init__(self, client: LLMClient):
        self.client = client
        self.last_prompt = ""
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []

    def answer(
        self,
        query: str,
        conversation: list[dict] | None = None,
        token_budget: int | None = None,
    ) -> str:
        context_started = time.perf_counter()
        conversation = conversation or []
        lines = [
            f"MEMORY[{i}] {turn.get('content', '')}"
            for i, turn in enumerate(conversation, start=1)
            if str(turn.get("content", "")).strip()
        ]
        header = (
            "You are a full-history baseline agent.\n"
            "Use the conversation history below to answer the question.\n"
            + ANSWER_FORMAT_INSTRUCTION
            + "\n"
        )
        indices = select_context_indices(lines, query, token_budget, header)
        selected = [lines[i] for i in indices]
        self.last_context = "\n".join(selected)
        self.last_retrieved_ids = []
        self.last_prompt = (
            header
            + self.last_context
            + "\n\n"
            + f"QUESTION: {query}\n"
        )
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        self.last_memory_latency_ms = 0.0
        self.last_context_latency_ms = (time.perf_counter() - context_started) * 1000
        response, self.last_llm_latency_ms = timed_generate(self.client, self.last_prompt)
        return response
