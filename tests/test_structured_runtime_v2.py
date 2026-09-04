from __future__ import annotations

from datetime import datetime, timezone

from agent import StructuredKeyValueAgent, StructuredMemoryRuntimeAgent
from memory import HashEmbeddingModel, MemoryRecord, MemoryRuntimeV1
from memory.storage import InMemoryMemoryStore


class CaptureClient:
    def __init__(self, answer: str = "FALLBACK") -> None:
        self.answer = answer
        self.calls = 0
        self.last_usage = {}
        self.last_attempts = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_attempts = 1
        return self.answer


class CountingEmbedding(HashEmbeddingModel):
    def __init__(self) -> None:
        super().__init__(dim=64)
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return super().encode(texts)


CONVERSATION = [
    {
        "memory_id": "fact-v1",
        "role": "user",
        "content": "极光记忆单元01数据库为 SQLite-01。",
        "valid_from": "2026-01-01T09:00:00+08:00",
        "valid_to": "2026-02-01T09:00:00+08:00",
    },
    {
        "memory_id": "fact-v2",
        "role": "user",
        "content": "极光记忆单元01数据库改为 GaussDB-01。",
        "valid_from": "2026-02-01T09:00:00+08:00",
    },
    {
        "memory_id": "noise-predicate",
        "role": "user",
        "content": "旁路单元01数据库为 辅助值-01。",
        "valid_from": "2026-03-01T09:00:00+08:00",
    },
    {
        "memory_id": "noise-subject",
        "role": "user",
        "content": "极光记忆单元01架构为 单体。",
        "valid_from": "2026-04-01T09:00:00+08:00",
    },
]
QUERY_TIME = "2026-05-01T09:00:00+08:00"


def test_structured_runtime_uses_prefilter_alias_and_fast_path_without_embedding_scan():
    embedder = CountingEmbedding()
    client = CaptureClient()
    runtime = MemoryRuntimeV1(embedder=embedder)
    agent = StructuredMemoryRuntimeAgent(client, runtime)
    agent.ingest(CONVERSATION, user_id="benchmark", session_id="test")
    embedder.calls = 0

    answer = agent.answer(
        "极光记忆单元01目前采用哪一种持久化后端？",
        query_time=QUERY_TIME,
    )

    assert answer == "GaussDB-01"
    assert client.calls == 0
    assert embedder.calls == 0
    assert agent.last_answer_route == "structured_fast_path"
    assert agent.last_retrieval_route == "exact_spo_prefilter"
    assert len(agent.last_retrieved_ids) == 1
    assert agent.last_retrieved_contents == ["极光记忆单元01数据库改为 GaussDB-01。"]
    assert "CURRENT_VALUE[GaussDB-01]" in agent.last_context
    assert "QUERY_TIME[2026-05-01T09:00:00+08:00]" in agent.last_prompt
    assert "VALID_FROM[" in agent.last_context
    assert "VALID_TO[" in agent.last_context


def test_structured_runtime_falls_back_when_fact_key_is_not_unique():
    client = CaptureClient("fallback-answer")
    agent = StructuredMemoryRuntimeAgent(
        client,
        MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=64)),
    )
    agent.ingest(CONVERSATION, user_id="benchmark", session_id="test")
    answer = agent.answer(
        "比较极光记忆单元01和旁路单元01的数据库。",
        query_time=QUERY_TIME,
    )
    assert answer == "fallback-answer"
    assert client.calls == 1
    assert agent.last_answer_route == "llm_fallback"
    assert agent.last_fast_path_reason == "fact_key_not_uniquely_resolved"


def test_structured_runtime_falls_back_for_two_visible_versions():
    store = InMemoryMemoryStore()
    point = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, value in enumerate(("A", "B"), start=1):
        store.add(MemoryRecord(
            memory_id=f"conflict-{index}",
            user_id="benchmark",
            content=f"系统数据库为 {value}。",
            created_at=point,
            valid_from=point,
            subject="系统",
            predicate="数据库",
            object_value=value,
            version=index,
        ))
    client = CaptureClient("conflict-fallback")
    agent = StructuredMemoryRuntimeAgent(
        client,
        MemoryRuntimeV1(store=store, embedder=HashEmbeddingModel(dim=64)),
    )
    agent.user_id = "benchmark"
    assert agent.answer("系统当前的数据库是什么？") == "conflict-fallback"
    assert client.calls == 1
    assert agent.last_fast_path_reason == "visible_version_not_unique"


def test_structured_kv_is_strong_alias_aware_control():
    agent = StructuredKeyValueAgent()
    agent.ingest(CONVERSATION, user_id="benchmark")
    answer = agent.answer(
        "极光记忆单元01当前的数据存储系统是什么？",
        query_time=QUERY_TIME,
    )
    assert answer == "GaussDB-01"
    assert agent.last_answer_route == "structured_fast_path"
    assert agent.last_llm_latency_ms == 0
    assert agent.last_retrieved_ids == ["fact-v2"]
