from __future__ import annotations

from .base import Agent, LLMClient, estimate_tokens


class NoMemoryAgent(Agent):
    """B0 baseline: only the current query is sent to the model."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.last_prompt = ""
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []

    def answer(self, query: str) -> str:
        self.last_context = ""
        self.last_retrieved_ids = []
        self.last_prompt = (
            "You are a baseline agent with no persistent memory.\n"
            "Answer only from the current question. If required information is absent, answer UNKNOWN.\n\n"
            f"QUESTION: {query}\n"
        )
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        return self.client.generate(self.last_prompt)
