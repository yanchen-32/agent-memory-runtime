from __future__ import annotations

from datetime import datetime
import time

from memory.runtime import MemoryRuntimeV1
from memory.schema import MemoryType, coerce_datetime

from .base import (
    ANSWER_FORMAT_INSTRUCTION,
    Agent,
    LLMClient,
    estimate_tokens,
    timed_generate,
)


class MemoryRuntimeAgent(Agent):
    """Benchmark adapter for the proposed Memory Runtime."""

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
        self.last_budget_selection = None
        self.last_trace_id: str | None = None

    def ingest(
        self,
        messages: list[dict],
        user_id: str = "default",
        session_id: str = "default",
        memory_type_override: MemoryType | None = None,
        preserve_duplicates: bool = False,
    ):
        self.user_id = user_id
        self.session_id = session_id
        outputs = []
        for message in messages:
            write_time = coerce_datetime(
                message.get("valid_from") or message.get("write_time")
            )
            outputs.extend(
                self.runtime.write(
                    [message],
                    user_id=user_id,
                    session_id=session_id,
                    at=write_time,
                    memory_type_override=memory_type_override,
                    preserve_duplicates=preserve_duplicates,
                )
            )
        return outputs

    def answer(
        self,
        query: str,
        token_budget: int | None = None,
        query_time: datetime | str | None = None,
        trace_id: str | None = None,
        query_id: str | None = None,
    ) -> str:
        memory_started = time.perf_counter()
        result = self.runtime.read(
            query,
            top_k=self.top_k,
            user_id=self.user_id,
            query_time=query_time,
            trace_id=trace_id,
            query_id=query_id,
        )
        self.last_memory_latency_ms = (time.perf_counter() - memory_started) * 1000
        context_started = time.perf_counter()
        header = (
            "You are an Agent Memory Runtime agent.\n"
            "Use only the retrieved memory below.\n"
            + ANSWER_FORMAT_INSTRUCTION
            + "\n"
        )
        suffix = "\n\n" + f"QUESTION: {query}\n"
        selection = self.runtime.select_context(
            query=query,
            result=result,
            token_budget=token_budget,
            prefix=header,
            suffix=suffix,
            trace_id=trace_id,
            query_id=query_id,
        )
        selected_hits = selection.selected
        selected_lines = [
            f"MEMORY[{i}] {hit.content}"
            for i, hit in enumerate(selected_hits, start=1)
        ]
        self.last_context = "\n".join(selected_lines)
        self.last_retrieved_ids = [hit.memory_id for hit in selected_hits]
        self.last_retrieved_contents = [hit.content for hit in selected_hits]
        self.last_hits = selected_hits
        self.last_budget_selection = selection
        self.last_trace_id = self.runtime.last_trace_id
        self.last_prompt = header + self.last_context + suffix
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        self.last_context_latency_ms = (time.perf_counter() - context_started) * 1000
        response, self.last_llm_latency_ms = timed_generate(self.client, self.last_prompt)
        return response
