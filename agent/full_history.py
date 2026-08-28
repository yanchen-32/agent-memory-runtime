from __future__ import annotations

from .base import Agent, LLMClient, estimate_tokens, select_context_indices


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
        conversation = conversation or []
        lines = [
            f"MEMORY[{i}] {turn.get('content', '')}"
            for i, turn in enumerate(conversation, start=1)
            if str(turn.get("content", "")).strip()
        ]
        header = (
            "You are a full-history baseline agent.\n"
            "Use the conversation history below to answer the question. "
            "If the history does not support an answer, answer UNKNOWN.\n\n"
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
        return self.client.generate(self.last_prompt)
