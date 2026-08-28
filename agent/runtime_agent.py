from __future__ import annotations

from memory.runtime import MemoryRuntimeV1

from .base import Agent, LLMClient, estimate_tokens, select_context_indices


class MemoryRuntimeAgent(Agent):
    """Benchmark adapter for the proposed Memory Runtime.

    The adapter keeps the runtime lifecycle outside the Agent interface so the
    same runtime can later be connected to a real application or service.
    """

    def __init__(self, client: LLMClient, runtime: MemoryRuntimeV1, top_k: int = 5):
        self.client = client
        self.runtime = runtime
        self.top_k = top_k
        self.user_id = "default"
        self.session_id = "default"
        self.last_prompt = ""
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []
        self.last_retrieved_contents: list[str] = []
        self.last_hits = []

    def ingest(
        self,
        messages: list[dict],
        user_id: str = "default",
        session_id: str = "default",
    ):
        self.user_id = user_id
        self.session_id = session_id
        return self.runtime.write(messages, user_id=user_id, session_id=session_id)

    def answer(self, query: str, token_budget: int | None = None) -> str:
        result = self.runtime.read(query, top_k=self.top_k, user_id=self.user_id)
        lines = [
            f"MEMORY[{i}] {hit.content}"
            for i, hit in enumerate(result.hits, start=1)
        ]
        header = (
            "You are an Agent Memory Runtime agent.\n"
            "Use only the retrieved memory below. "
            "If the memory does not support an answer, answer UNKNOWN.\n\n"
        )
        indices = select_context_indices(lines, query, token_budget, header)
        selected_hits = [result.hits[i] for i in indices]
        selected_lines = [lines[i] for i in indices]
        self.last_context = "\n".join(selected_lines)
        self.last_retrieved_ids = [hit.memory_id for hit in selected_hits]
        self.last_retrieved_contents = [hit.content for hit in selected_hits]
        self.last_hits = selected_hits
        self.last_prompt = (
            header
            + self.last_context
            + "\n\n"
            + f"QUESTION: {query}\n"
        )
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        return self.client.generate(self.last_prompt)
