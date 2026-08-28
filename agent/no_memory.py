from __future__ import annotations

from .base import Agent, LLMClient


class NoMemoryAgent(Agent):
    """B0 baseline: only the current query is sent to the model."""

    def __init__(self, client: LLMClient):
        self.client = client

    def answer(self, query: str) -> str:
        prompt = (
            "You are a baseline agent with no persistent memory.\n"
            "Answer only from the current question. If required information is absent, answer UNKNOWN.\n\n"
            f"QUESTION: {query}\n"
        )
        return self.client.generate(prompt)
