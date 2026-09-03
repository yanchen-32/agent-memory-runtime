"""Development-only E3 prompt and retrieval ablations.

These variants are deliberately isolated from ``MemoryRuntimeAgent`` so the
frozen v1.3-E3 implementation and its recorded tooling hashes stay unchanged.
Each variant adds exactly one behavior to the preceding variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

from memory.runtime import MemoryRuntimeV1
from memory.schema import ReadResult

from .base import (
    ANSWER_FORMAT_INSTRUCTION,
    estimate_tokens,
    memory_line,
    timed_generate,
)
from .runtime_agent import MemoryRuntimeAgent


E3_ABLATION_PROMPT_VERSION = "e3-development-ablation-v1"
E3_VERSION_TIME_INSTRUCTION = (
    "For this versioned current-state question, answer at QUERY_TIME. "
    "A memory applies over [VALID_FROM, VALID_TO); "
    "VALID_TO=UNSPECIFIED means it remains current at QUERY_TIME.\n"
)


@dataclass(frozen=True, slots=True)
class E3AblationSpec:
    name: str
    canonical_current_fact: bool
    version_time_context: bool
    exact_spo_filter: bool
    identity_title: str
    changed_variable: str


DEFAULT_IDENTITY_TITLE = "You are an Agent Memory Runtime agent."
B2_IDENTITY_TITLE = "You are a vector-memory baseline agent."

E3_ABLATION_SPECS = {
    "A0": E3AblationSpec(
        "A0", False, False, False, DEFAULT_IDENTITY_TITLE, "frozen_ours_control"
    ),
    "A1": E3AblationSpec(
        "A1", True, False, False, DEFAULT_IDENTITY_TITLE, "canonical_current_fact"
    ),
    "A2": E3AblationSpec(
        "A2", True, True, False, DEFAULT_IDENTITY_TITLE, "version_time_context"
    ),
    "A3": E3AblationSpec(
        "A3", True, True, True, DEFAULT_IDENTITY_TITLE, "exact_spo_filter"
    ),
    "A4": E3AblationSpec(
        "A4", True, True, True, B2_IDENTITY_TITLE, "identity_title"
    ),
}


class E3AblationAgent(MemoryRuntimeAgent):
    """Ours with cumulative, predeclared E3 Development ablation switches."""

    def __init__(
        self,
        client,
        runtime: MemoryRuntimeV1,
        *,
        variant: str,
        top_k: int = 5,
    ):
        if variant not in E3_ABLATION_SPECS:
            raise ValueError(f"unknown E3 ablation variant: {variant}")
        super().__init__(client, runtime, top_k=top_k)
        self.variant = variant
        self.spec = E3_ABLATION_SPECS[variant]
        self.last_spo_filtered_count = 0

    @staticmethod
    def _canonical_content(record, fallback: str) -> str:
        if (
            record is not None
            and record.version > 1
            and record.subject
            and record.predicate
            and record.object_value is not None
        ):
            return (
                f"CURRENT_FACT SUBJECT[{record.subject}] "
                f"PREDICATE[{record.predicate}] "
                f"CURRENT_VALUE[{record.object_value}]"
            )
        return fallback

    def _filter_exact_spo(self, query: str, result: ReadResult) -> ReadResult:
        if not self.spec.exact_spo_filter:
            self.last_spo_filtered_count = 0
            return result
        selected = []
        for hit in result.hits:
            record = self.runtime.store.get(hit.memory_id)
            if (
                record is not None
                and record.subject
                and record.predicate
                and record.subject in query
                and record.predicate in query
            ):
                selected.append(hit)
        self.last_spo_filtered_count = len(result.hits) - len(selected)
        return ReadResult(
            query=result.query,
            hits=selected,
            context="",
            query_time=result.query_time,
        )

    def answer(
        self,
        query: str,
        token_budget: int | None = None,
        query_time: datetime | str | None = None,
        trace_id: str | None = None,
        query_id: str | None = None,
        temporal_context: bool = False,
    ) -> str:
        # A0 delegates to the frozen implementation byte-for-byte at the
        # interface boundary and is the within-run control.
        if self.variant == "A0":
            return super().answer(
                query,
                token_budget=token_budget,
                query_time=query_time,
                trace_id=trace_id,
                query_id=query_id,
                temporal_context=False,
            )

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
        result = self._filter_exact_spo(query, result)
        time_header = (
            E3_VERSION_TIME_INSTRUCTION
            + f"QUERY_TIME[{query_time if query_time is not None else 'UNSPECIFIED'}]\n"
            if self.spec.version_time_context
            else ""
        )
        header = (
            self.spec.identity_title
            + "\nUse only the retrieved memory below.\n"
            + time_header
            + ANSWER_FORMAT_INSTRUCTION
            + "\n"
        )
        suffix = "\n\n" + f"QUESTION: {query}\n"
        context_lines = []
        for index, hit in enumerate(result.hits, start=1):
            record = self.runtime.store.get(hit.memory_id)
            content = (
                self._canonical_content(record, hit.content)
                if self.spec.canonical_current_fact
                else hit.content
            )
            context_lines.append(
                memory_line(
                    index,
                    content,
                    temporal_context=self.spec.version_time_context,
                    valid_from=record.valid_from if record is not None else None,
                    valid_to=record.valid_to if record is not None else None,
                )
            )
        selection = self.runtime.budget_manager.select(
            query=query,
            candidates=result.hits,
            context_lines=context_lines,
            token_budget=token_budget,
            prefix=header,
            suffix=suffix,
        )
        selected_hits = selection.selected
        selected_lines = [context_lines[index] for index in selection.selected_indices]
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
