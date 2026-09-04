from __future__ import annotations

from datetime import datetime
import time

from memory.runtime import MemoryRuntimeV1
from memory.schema import ReadResult

from .base import ANSWER_FORMAT_INSTRUCTION, estimate_tokens, memory_line, timed_generate
from .runtime_agent import MemoryRuntimeAgent


MEMORY_RUNTIME_METHOD_VERSION = "memory-runtime-v2-structured-fast-path"
MEMORY_RUNTIME_PROMPT_VERSION = "structured-temporal-evidence-v2"
VERSION_TIME_INSTRUCTION = (
    "For this versioned fact question, answer at QUERY_TIME. "
    "A memory applies over [VALID_FROM, VALID_TO); "
    "VALID_TO=UNSPECIFIED means it remains valid at QUERY_TIME.\n"
)


class StructuredMemoryRuntimeAgent(MemoryRuntimeAgent):
    """A3 quality fixes plus an auditable, constrained structured fast path."""

    def __init__(
        self,
        client,
        runtime: MemoryRuntimeV1,
        top_k: int = 5,
        fast_path_min_confidence: float = 0.5,
    ) -> None:
        super().__init__(client, runtime, top_k=top_k)
        self.fast_path_min_confidence = fast_path_min_confidence
        self.method_version = MEMORY_RUNTIME_METHOD_VERSION
        self.prompt_version = MEMORY_RUNTIME_PROMPT_VERSION
        self.last_answer_route = "not_run"
        self.last_retrieval_route = "not_run"
        self.last_fast_path_eligible = False
        self.last_fast_path_reason = "not_run"
        self.last_fast_path_latency_ms = 0.0
        self.last_llm_fallback_latency_ms = 0.0
        self.last_resolved_subject: str | None = None
        self.last_resolved_predicate: str | None = None

    @staticmethod
    def _canonical_content(record, fallback: str) -> str:
        if (
            record is not None
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

    def _fast_path_decision(self, selected_hits) -> tuple[bool, str, object | None]:
        resolution = self.runtime.exact_fact_retriever.last_resolution
        if resolution is None:
            return False, "fact_key_not_uniquely_resolved", None
        if self.runtime.exact_fact_retriever.last_visible_record_count != 1:
            return False, "visible_version_not_unique", None
        if len(selected_hits) != 1:
            return False, "selected_evidence_not_unique", None
        record = self.runtime.store.get(selected_hits[0].memory_id)
        if record is None or record.object_value is None:
            return False, "structured_object_missing", record
        if record.confidence < self.fast_path_min_confidence:
            return False, "confidence_below_threshold", record
        if bool(record.metadata.get("unresolved_conflict")):
            return False, "unresolved_conflict", record
        return True, "unique_valid_structured_fact", record

    def answer(
        self,
        query: str,
        token_budget: int | None = None,
        query_time: datetime | str | None = None,
        trace_id: str | None = None,
        query_id: str | None = None,
        temporal_context: bool = True,
    ) -> str:
        answer_started = time.perf_counter()
        memory_started = time.perf_counter()
        exact_hits = self.runtime.exact_fact_retriever.search(
            query,
            top_k=self.top_k,
            user_id=self.user_id,
            query_time=query_time,
        )
        if self.runtime.exact_fact_retriever.last_resolution is not None:
            result = ReadResult(
                query=query,
                hits=exact_hits,
                context="",
                query_time=query_time,
            )
            self.last_retrieval_route = "exact_spo_prefilter"
        else:
            result = self.runtime.read(
                query,
                top_k=self.top_k,
                user_id=self.user_id,
                query_time=query_time,
                trace_id=trace_id,
                query_id=query_id,
            )
            self.last_retrieval_route = "hybrid_fallback"
        self.last_memory_latency_ms = (time.perf_counter() - memory_started) * 1000

        resolution = self.runtime.exact_fact_retriever.last_resolution
        self.last_resolved_subject = resolution.subject if resolution else None
        self.last_resolved_predicate = resolution.predicate if resolution else None
        use_temporal_context = temporal_context or query_time is not None
        context_started = time.perf_counter()
        header = (
            "You are an Agent Memory Runtime v2 agent.\n"
            "Use only the governed memory evidence below.\n"
            + (
                VERSION_TIME_INSTRUCTION
                + f"QUERY_TIME[{query_time if query_time is not None else 'UNSPECIFIED'}]\n"
                if use_temporal_context
                else ""
            )
            + ANSWER_FORMAT_INSTRUCTION
            + "\n"
        )
        suffix = "\n\n" + f"QUESTION: {query}\n"
        context_lines = []
        for index, hit in enumerate(result.hits, start=1):
            record = self.runtime.store.get(hit.memory_id)
            context_lines.append(memory_line(
                index,
                self._canonical_content(record, hit.content),
                temporal_context=use_temporal_context,
                valid_from=record.valid_from if record is not None else None,
                valid_to=record.valid_to if record is not None else None,
            ))
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

        eligible, reason, record = self._fast_path_decision(selected_hits)
        self.last_fast_path_eligible = eligible
        self.last_fast_path_reason = reason
        if eligible:
            self.last_answer_route = "structured_fast_path"
            self.last_llm_latency_ms = 0.0
            self.last_llm_fallback_latency_ms = 0.0
            self.last_fast_path_latency_ms = (time.perf_counter() - answer_started) * 1000
            return str(record.object_value)

        self.last_answer_route = "llm_fallback"
        self.last_fast_path_latency_ms = 0.0
        response, self.last_llm_latency_ms = timed_generate(self.client, self.last_prompt)
        self.last_llm_fallback_latency_ms = self.last_llm_latency_ms
        return response
