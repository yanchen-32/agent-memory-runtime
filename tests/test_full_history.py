from agent import FullHistoryAgent


class CaptureClient:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "ok"


def test_full_history_sends_history_without_retrieval():
    client = CaptureClient()
    agent = FullHistoryAgent(client)
    result = agent.answer(
        "项目数据库是什么？",
        conversation=[
            {"memory_id": "m1", "role": "user", "content": "项目数据库使用 SQLite。"},
            {"memory_id": "m2", "role": "user", "content": "项目部署平台是 openEuler。"},
        ],
    )
    assert result == "ok"
    assert "项目数据库使用 SQLite。" in client.prompts[0]
    assert "项目部署平台是 openEuler。" in client.prompts[0]
    assert agent.last_retrieved_ids == []
    assert agent.last_context.count("MEMORY[") == 2


def test_full_history_preserves_available_timestamps():
    client = CaptureClient()
    agent = FullHistoryAgent(client)
    agent.answer(
        "历史状态？",
        conversation=[{
            "content": "项目数据库使用 SQLite。",
            "valid_from": "2026-01-01T00:00:00+08:00",
        }],
    )
    assert "TIME[2026-01-01T00:00:00+08:00]" in agent.last_context
