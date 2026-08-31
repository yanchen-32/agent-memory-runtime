from __future__ import annotations

import time

from .base import (
    ANSWER_FORMAT_INSTRUCTION,
    Agent,
    LLMClient,
    estimate_tokens,
    timed_generate,
)


class NoMemoryAgent(Agent):
    """B0 baseline: only the current query is sent to the model."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.last_prompt = ""
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []

    def answer(self, query: str) -> str:
        context_started = time.perf_counter()
        self.last_context = ""
        self.last_retrieved_ids = []
        self.last_prompt = (
            "You are a baseline agent with no persistent memory.\n"
            "Answer only from the current question.\n"
            + ANSWER_FORMAT_INSTRUCTION
            + "\n"
            f"QUESTION: {query}\n"
        )
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        self.last_memory_latency_ms = 0.0
        self.last_context_latency_ms = (time.perf_counter() - context_started) * 1000
        response, self.last_llm_latency_ms = timed_generate(self.client, self.last_prompt)
        return response
