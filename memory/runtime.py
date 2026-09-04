from __future__ import annotations

from datetime import datetime
import time

from memory.context_budget import BudgetSelection, ContextBudgetManager
from memory.consolidation import (
    AdaptiveConsolidationPolicy,
    ConsolidationReport,
    MemoryConsolidator,
)
from memory.embedding import EmbeddingModel, HashEmbeddingModel
from memory.governance import Deduplicator
from memory.lifecycle import ForgettingPolicy, MemoryCompressor
from memory.reranker import WeightedReranker
from memory.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from memory.schema import MemoryType, ReadResult, coerce_datetime
from memory.observability import TraceEvent, TraceRecorder
from memory.context_budget import estimate_tokens
from memory.service import MemoryReaderV1, MemoryWriterV1
from memory.storage import InMemoryMemoryStore, MemoryStore
from memory.structured_lookup import ExactFactRetriever, StructuredFactResolver


class MemoryRuntimeV1:
    """Convenience facade wiring memory, temporal and budget components."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedder: EmbeddingModel | None = None,
        budget_manager: ContextBudgetManager | None = None,
        consolidation_policy: AdaptiveConsolidationPolicy | None = None,
        trace_enabled: bool = False,
        structured_resolver: StructuredFactResolver | None = None,
    ):
        self.store = store or InMemoryMemoryStore()
        self.embedder = embedder or HashEmbeddingModel(dim=384)

        self.vector_retriever = VectorRetriever(self.store, self.embedder)
        self.bm25_retriever = BM25Retriever(self.store)
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.bm25_retriever)
        self.exact_fact_retriever = ExactFactRetriever(
            self.store,
            resolver=structured_resolver,
        )
        self.reranker = WeightedReranker(self.store)
        self.compressor = MemoryCompressor()
        self.deduplicator = Deduplicator(self.embedder)

        self.writer = MemoryWriterV1(self.store, self.deduplicator)
        self.reader = MemoryReaderV1(self.store, self.hybrid_retriever, self.reranker, self.compressor)
        self.forgetting = ForgettingPolicy(self.store)
        self.budget_manager = budget_manager or ContextBudgetManager(self.store)
        self.consolidator = MemoryConsolidator(self.store)
        self.adaptive_consolidator = MemoryConsolidator(
            self.store,
            policy=consolidation_policy or AdaptiveConsolidationPolicy(),
        )
        self.trace_recorder = TraceRecorder(enabled=trace_enabled)
        self.last_trace_id: str | None = None

    def write(
        self,
        messages,
        user_id: str = "default",
        session_id: str = "default",
        at: datetime | str | None = None,
        memory_type_override: MemoryType | None = None,
        preserve_duplicates: bool = False,
    ):
        return self.writer.write(
            messages,
            user_id=user_id,
            session_id=session_id,
            at=at,
            memory_type_override=memory_type_override,
            preserve_duplicates=preserve_duplicates,
        )

    def read(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
        trace_id: str | None = None,
        query_id: str | None = None,
    ) -> ReadResult:
        if self.trace_recorder.enabled:
            trace_id = trace_id or self.trace_recorder.new_trace_id()
            self.last_trace_id = trace_id
        else:
            self.last_trace_id = None
        started = time.perf_counter()
        result = self.reader.read(
            query,
            top_k=top_k,
            user_id=user_id,
            query_time=coerce_datetime(query_time),
        )
        if self.trace_recorder.enabled and trace_id is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            selected_ids = {hit.memory_id for hit in result.hits}
            for rank, hit in enumerate(self.reader.last_candidates, start=1):
                record = self.store.get(hit.memory_id)
                features = hit.metadata.get("features", {})
                self.trace_recorder.record(TraceEvent(
                    trace_id=trace_id,
                    query_id=query_id,
                    stage="retrieval_rerank",
                    memory_id=hit.memory_id,
                    version=record.version if record else None,
                    source_ids=list(record.source_ids) if record else [],
                    score_components={
                        "semantic_score": hit.metadata.get("vector_score"),
                        "bm25_score": hit.metadata.get("bm25_score"),
                        "rrf_score": hit.score,
                        "recency_score": features.get("recency"),
                        "importance_score": features.get("importance"),
                        "confidence": record.confidence if record else None,
                        "reranker_score": next(
                            (value.score for value in result.hits if value.memory_id == hit.memory_id),
                            None,
                        ),
                    },
                    rank=rank,
                    selected=hit.memory_id in selected_ids,
                    reject_reason=None if hit.memory_id in selected_ids else "low_score",
                    token_cost=estimate_tokens(hit.content),
                    latency_ms=latency_ms,
                ))
            if query_time is None:
                for record in self.store.list_all(user_id=user_id):
                    if record.status.value == "superseded":
                        self.trace_recorder.record(TraceEvent(
                            trace_id=trace_id,
                            query_id=query_id,
                            stage="temporal_filter",
                            memory_id=record.memory_id,
                            version=record.version,
                            source_ids=list(record.source_ids),
                            selected=False,
                            reject_reason="expired_version",
                        ))
        return result

    def select_context(
        self,
        query: str,
        result: ReadResult,
        token_budget: int | None,
        prefix: str = "",
        suffix: str = "",
        trace_id: str | None = None,
        query_id: str | None = None,
    ) -> BudgetSelection:
        lines = [
            f"MEMORY[{i}] {hit.content}"
            for i, hit in enumerate(result.hits, start=1)
        ]
        started = time.perf_counter()
        selection = self.budget_manager.select(
            query=query,
            candidates=result.hits,
            context_lines=lines,
            token_budget=token_budget,
            prefix=prefix,
            suffix=suffix,
        )
        active_trace_id = trace_id or self.last_trace_id
        if self.trace_recorder.enabled and active_trace_id is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            selected_ids = {hit.memory_id for hit in selection.selected}
            for rank, hit in enumerate(result.hits, start=1):
                record = self.store.get(hit.memory_id)
                index = rank - 1
                details = selection.score_details.get(index, {})
                selected = hit.memory_id in selected_ids
                self.trace_recorder.record(TraceEvent(
                    trace_id=active_trace_id,
                    query_id=query_id,
                    stage="context_budget",
                    memory_id=hit.memory_id,
                    version=record.version if record else None,
                    source_ids=list(record.source_ids) if record else [],
                    score_components=details,
                    rank=rank,
                    selected=selected,
                    reject_reason=None if selected else "budget_exceeded",
                    token_cost=estimate_tokens(lines[index]),
                    latency_ms=latency_ms,
                    metadata={
                        "tokens_before": selection.tokens_before,
                        "tokens_after": selection.tokens_after,
                        "budget": selection.budget,
                    },
                ))
        return selection

    def consolidate(
        self,
        user_id: str | None = None,
        strategy: str = "fixed",
        now: datetime | None = None,
        trace_id: str | None = None,
    ) -> ConsolidationReport:
        """Consolidate active episodic memories and keep source traceability."""
        if strategy == "fixed":
            report = self.consolidator.consolidate(user_id=user_id, now=now)
        elif strategy == "adaptive":
            report = self.adaptive_consolidator.consolidate(user_id=user_id, now=now)
        else:
            raise ValueError("strategy must be 'fixed' or 'adaptive'")
        if self.trace_recorder.enabled:
            trace_id = trace_id or self.trace_recorder.new_trace_id()
            self.last_trace_id = trace_id
            for group in report.groups:
                self.trace_recorder.record(TraceEvent(
                    trace_id=trace_id,
                    query_id=None,
                    stage="consolidation",
                    memory_id=group.semantic_memory_id,
                    source_ids=list(group.source_ids),
                    selected=True,
                    token_cost=group.tokens_after,
                    score_components={
                        "trigger_score": group.trigger_score,
                        "granularity_score": group.granularity_score,
                    },
                    metadata={
                        "granularity_level": group.granularity_level,
                        "policy_version": group.policy_version,
                        "tokens_before": group.tokens_before,
                        "tokens_after": group.tokens_after,
                        "compression_ratio": group.compression_ratio,
                    },
                ))
        return report

    def trace(
        self,
        trace_id: str | None = None,
        memory_id: str | None = None,
        version_group: str | None = None,
    ) -> list[dict]:
        """Read mechanistic events or a memory evolution timeline."""
        if trace_id is not None:
            return self.trace_recorder.get(trace_id)
        records = self.store.list_all()
        if memory_id is not None:
            target = self.store.get(memory_id)
            if target is None:
                return []
            version_group = target.version_group
        if version_group is not None:
            return [
                record.to_dict()
                for record in records
                if record.version_group == version_group
            ]
        return self.trace_recorder.all()
