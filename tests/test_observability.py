from datetime import datetime, timezone

from agent import MemoryRuntimeAgent, RuleBasedClient
from memory import HashEmbeddingModel, MemoryRuntimeV1, MemoryType


def test_runtime_trace_records_retrieval_temporal_and_budget_decisions():
    runtime = MemoryRuntimeV1(
        embedder=HashEmbeddingModel(dim=64),
        trace_enabled=True,
    )
    old = runtime.write(
        [{"role": "user", "content": "项目数据库是 SQLite。"}],
        user_id="u1",
        at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )[0]
    new = runtime.write(
        [{"role": "user", "content": "项目数据库改为 openGauss。"}],
        user_id="u1",
        at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )[0]
    agent = MemoryRuntimeAgent(RuleBasedClient(), runtime)
    agent.user_id = "u1"
    agent.answer("项目数据库是什么？", token_budget=80, query_id="query-1")

    events = runtime.trace(trace_id=agent.last_trace_id)
    assert {event["stage"] for event in events} >= {
        "retrieval_rerank", "temporal_filter", "context_budget"
    }
    expired = [event for event in events if event["reject_reason"] == "expired_version"]
    assert expired and expired[0]["memory_id"] == old.memory_id
    budget = [event for event in events if event["stage"] == "context_budget"]
    assert budget[0]["selected"] is True

    timeline = runtime.trace(memory_id=new.memory_id)
    assert [record["version"] for record in timeline] == [1, 2]


def test_consolidation_trace_contains_real_lineage():
    runtime = MemoryRuntimeV1(
        embedder=HashEmbeddingModel(dim=64),
        trace_enabled=True,
    )
    for content in (
        "项目数据库使用 openGauss。",
        "项目数据库后来仍使用 openGauss。",
    ):
        runtime.write(
            [{"role": "user", "content": content}],
            user_id="u1",
            memory_type_override=MemoryType.EPISODIC,
            preserve_duplicates=True,
        )
    report = runtime.consolidate(user_id="u1", strategy="adaptive")
    events = runtime.trace(trace_id=runtime.last_trace_id)

    assert report.source_count == 2
    assert len(events) == 1
    assert events[0]["stage"] == "consolidation"
    assert len(events[0]["source_ids"]) == 2
    assert events[0]["metadata"]["policy_version"] == "adaptive-rule-v1"


def test_trace_disabled_is_zero_storage_default():
    runtime = MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=32))
    runtime.write([{"role": "user", "content": "项目数据库是 SQLite。"}])
    runtime.read("项目数据库是什么？")
    assert runtime.last_trace_id is None
    assert runtime.trace() == []
