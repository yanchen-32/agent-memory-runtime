from agent import (
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
