from agent import HybridMemoryAgent, RuleBasedClient
from memory import HashEmbeddingModel


def test_b3_hybrid_agent_runs_as_independent_baseline():
    agent = HybridMemoryAgent(
        RuleBasedClient(),
        embedder=HashEmbeddingModel(dim=64),
        top_k=3,
    )
    agent.ingest(
        [
            {"memory_id": "b3_exact", "content": "项目数据库使用 openGauss"},
            {"memory_id": "b3_noise", "content": "今天整理答辩材料"},
        ]
    )
    answer = agent.answer("项目数据库使用什么？")

    assert answer == "openGauss"
    assert agent.last_retrieved_ids
    assert "hybrid-memory baseline" in agent.last_prompt
    assert all(record.metadata["baseline"] == "B3" for record in agent.store.list_all())
