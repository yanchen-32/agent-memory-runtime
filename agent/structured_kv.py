from __future__ import annotations

from datetime import datetime
import time

from memory.extraction import RuleMemoryExtractor
from memory.schema import MemoryRecord, MemoryStatus, coerce_datetime
from memory.storage import InMemoryMemoryStore
from memory.structured_lookup import ExactFactRetriever, StructuredFactResolver

from .base import Agent, estimate_tokens, memory_line


STRUCTURED_KV_BASELINE_VERSION = "structured-kv-v1"


class StructuredKeyValueAgent(Agent):
    """Strong E3 control: last-valid structured fact lookup, without LLM or RAG."""

    def __init__(self, resolver: StructuredFactResolver | None = None) -> None:
        self.store = InMemoryMemoryStore()
        self.extractor = RuleMemoryExtractor()
        self.retriever = ExactFactRetriever(self.store, resolver=resolver)
        self.user_id = "default"
        self.last_prompt = ""
        self.last_prompt_tokens = 0
        self.last_context = ""
        self.last_retrieved_ids: list[str] = []
        self.last_retrieved_contents: list[str] = []
        self.last_memory_latency_ms = 0.0
        self.last_context_latency_ms = 0.0
        self.last_llm_latency_ms = 0.0
        self.last_answer_route = "not_run"
        self.last_retrieval_route = "exact_spo_prefilter"
        self.last_fast_path_eligible = False
        self.last_fast_path_reason = "not_run"
        self.last_fast_path_latency_ms = 0.0
        self.last_llm_fallback_latency_ms = 0.0
        self.method_version = STRUCTURED_KV_BASELINE_VERSION

    def ingest(self, messages: list[dict], user_id: str = "default", **_: object) -> None:
        self.user_id = user_id
        versions: dict[tuple[str, str], int] = {}
        for message in messages:
            candidates = self.extractor.extract([message])
            for candidate_index, candidate in enumerate(candidates):
                if not candidate.subject or not candidate.predicate:
                    continue
                key = (candidate.subject, candidate.predicate)
                versions[key] = versions.get(key, 0) + 1
                valid_from = coerce_datetime(
                    message.get("valid_from") or message.get("created_at")
                )
                if valid_from is None:
                    raise ValueError("StructuredKeyValueAgent requires valid_from")
                valid_to = coerce_datetime(message.get("valid_to"))
                memory_id = str(message.get("memory_id") or f"kv-{len(self.store.list_all())}")
                if len(candidates) > 1:
                    memory_id = f"{memory_id}-{candidate_index}"
                self.store.add(MemoryRecord(
                    memory_id=memory_id,
                    user_id=user_id,
                    content=candidate.content,
                    entities=candidate.entities,
                    keywords=candidate.keywords,
                    created_at=valid_from,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    status=(MemoryStatus.SUPERSEDED if valid_to is not None else MemoryStatus.ACTIVE),
                    version=versions[key],
                    subject=candidate.subject,
                    predicate=candidate.predicate,
                    object_value=candidate.object_value,
                    confidence=candidate.confidence,
                    metadata={"baseline": "StructuredKV"},
                ))

    def answer(
        self,
        query: str,
        query_time: datetime | str | None = None,
        **_: object,
    ) -> str:
        started = time.perf_counter()
        memory_started = time.perf_counter()
        hits = self.retriever.search(
            query,
            top_k=2,
            user_id=self.user_id,
            query_time=query_time,
        )
        self.last_memory_latency_ms = (time.perf_counter() - memory_started) * 1000
        context_started = time.perf_counter()
        self.last_retrieved_ids = [hit.memory_id for hit in hits]
        self.last_retrieved_contents = [hit.content for hit in hits]
        lines = []
        for index, hit in enumerate(hits, start=1):
            record = self.store.get(hit.memory_id)
            content = (
                f"CURRENT_FACT SUBJECT[{record.subject}] PREDICATE[{record.predicate}] "
                f"CURRENT_VALUE[{record.object_value}]"
                if record is not None and record.object_value is not None
                else hit.content
            )
            lines.append(memory_line(
                index,
                content,
                temporal_context=True,
                valid_from=record.valid_from if record else None,
                valid_to=record.valid_to if record else None,
            ))
        self.last_context = "\n".join(lines)
        self.last_prompt = self.last_context + f"\nQUESTION: {query}\n"
        self.last_prompt_tokens = estimate_tokens(self.last_prompt)
        self.last_context_latency_ms = (time.perf_counter() - context_started) * 1000
        resolution = self.retriever.last_resolution
        record = self.store.get(hits[0].memory_id) if len(hits) == 1 else None
        eligible = bool(
            resolution is not None
            and self.retriever.last_visible_record_count == 1
            and record is not None
            and record.object_value is not None
            and not record.metadata.get("unresolved_conflict")
        )
        self.last_fast_path_eligible = eligible
        self.last_fast_path_reason = (
            "unique_valid_structured_fact" if eligible else "fact_not_uniquely_resolved"
        )
        self.last_answer_route = "structured_fast_path" if eligible else "no_answer"
        self.last_fast_path_latency_ms = (time.perf_counter() - started) * 1000
        return str(record.object_value) if eligible else "UNKNOWN"
