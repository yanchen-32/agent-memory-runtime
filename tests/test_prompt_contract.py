from agent import (
    ANSWER_FORMAT_VERSION,
    FullHistoryAgent,
    HybridMemoryAgent,
    MemoryRuntimeAgent,
    NoMemoryAgent,
    VectorMemoryAgent,
)
from agent.base import ANSWER_FORMAT_INSTRUCTION
from memory import HashEmbeddingModel, MemoryRuntimeV1, VectorMemoryStore


class CaptureClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "UNKNOWN"


def test_all_agents_use_the_same_answer_format_contract():
    clients = [CaptureClient() for _ in range(5)]

    b0 = NoMemoryAgent(clients[0])
    b0.answer("问题？")

    b1 = FullHistoryAgent(clients[1])
    b1.answer("问题？", conversation=[{"content": "记忆内容"}])

    b2_store = VectorMemoryStore(HashEmbeddingModel(dim=32))
    b2_store.add("记忆内容")
    b2 = VectorMemoryAgent(clients[2], b2_store)
    b2.answer("问题？")

    b3 = HybridMemoryAgent(clients[3], embedder=HashEmbeddingModel(dim=32))
    b3.ingest([{"content": "记忆内容"}])
    b3.answer("问题？")

    ours = MemoryRuntimeAgent(
        clients[4], MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=32))
    )
    ours.ingest([{"role": "user", "content": "记忆内容"}])
    ours.answer("问题？")

    assert all(
        client.prompts[0].count(ANSWER_FORMAT_INSTRUCTION) == 1
        for client in clients
    )
    assert ANSWER_FORMAT_VERSION == "shortest-answer-v2-temporal"


def test_memory_agents_share_historical_prompt_contract_without_baseline_filtering():
    query_time = "2026-01-15T09:00:00+08:00"
    conversation = [
        {
            "memory_id": "db-v1",
            "role": "user",
            "content": "项目数据库使用 SQLite。",
            "valid_from": "2026-01-01T09:00:00+08:00",
        },
        {
            "memory_id": "db-v2",
            "role": "user",
            "content": "项目数据库改为 openGauss。",
            "valid_from": "2026-02-01T09:00:00+08:00",
        },
    ]
    clients = [CaptureClient() for _ in range(4)]

    b1 = FullHistoryAgent(clients[0])
    b1.answer(
        "2026年1月15日项目使用什么数据库？",
        conversation=conversation,
        query_time=query_time,
        temporal_context=True,
    )

    b2_store = VectorMemoryStore(HashEmbeddingModel(dim=32))
    b2_store.add_many(
        [turn["content"] for turn in conversation],
        memory_ids=[turn["memory_id"] for turn in conversation],
        metadata=[
            {"valid_from": turn["valid_from"], "valid_to": turn.get("valid_to")}
            for turn in conversation
        ],
    )
    b2 = VectorMemoryAgent(clients[1], b2_store)
    b2.answer(
        "2026年1月15日项目使用什么数据库？",
        query_time=query_time,
        temporal_context=True,
    )

    b3 = HybridMemoryAgent(clients[2], embedder=HashEmbeddingModel(dim=32))
    b3.ingest(conversation)
    b3.answer(
        "2026年1月15日项目使用什么数据库？",
        query_time=query_time,
        temporal_context=True,
    )

    ours = MemoryRuntimeAgent(
        clients[3], MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=32))
    )
    ours.ingest(conversation)
    ours.answer(
        "2026年1月15日项目使用什么数据库？",
        query_time=query_time,
        temporal_context=True,
    )

    for client in clients:
        prompt = client.prompts[0]
        assert f"QUERY_TIME[{query_time}]" in prompt
        assert "VALID_FROM[" in prompt
        assert "VALID_TO[" in prompt
    assert set(b2.last_retrieved_ids) == {"db-v1", "db-v2"}
    assert set(b3.last_retrieved_ids) == {"db-v1", "db-v2"}
    assert "SQLite" in ours.last_context
    assert "openGauss" not in ours.last_context
    assert "VALID_TO[2026-02-01T09:00:00+08:00]" in ours.last_context
