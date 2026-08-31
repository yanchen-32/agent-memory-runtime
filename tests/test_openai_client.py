import json

from agent import OpenAICompatibleClient
import agent.clients as clients_module


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "  连接成功  "}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }).encode()


def test_openai_client_sends_frozen_generation_config(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(clients_module.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/",
        api_key="test-only",
        timeout=12,
        temperature=0,
        top_p=1,
        max_tokens=64,
        thinking="disabled",
    )
    assert client.generate("只回答：连接成功") == "连接成功"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 64
    assert client.last_usage["total_tokens"] == 6
