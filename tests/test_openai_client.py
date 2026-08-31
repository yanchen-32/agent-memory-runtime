import json
from urllib import error

import pytest

from agent import LLMRequestError, OpenAICompatibleClient
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
    assert client.last_attempts == 1


def test_openai_client_retries_retryable_http_error(monkeypatch):
    calls = 0

    def flaky_urlopen(req, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error.HTTPError(req.full_url, 429, "rate limited", None, None)
        return _Response()

    monkeypatch.setattr(clients_module.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(clients_module.time, "sleep", lambda _: None)
    client = OpenAICompatibleClient(
        model="model",
        base_url="https://example.invalid",
        api_key="test-only",
        max_retries=2,
        retry_backoff_seconds=0,
    )
    assert client.generate("test") == "连接成功"
    assert calls == 2
    assert client.last_attempts == 2


def test_openai_client_does_not_retry_authentication_error(monkeypatch):
    calls = 0

    def unauthorized(req, timeout):
        nonlocal calls
        calls += 1
        raise error.HTTPError(req.full_url, 401, "unauthorized", None, None)

    monkeypatch.setattr(clients_module.request, "urlopen", unauthorized)
    client = OpenAICompatibleClient(
        model="model",
        base_url="https://example.invalid",
        api_key="bad-test-key",
        max_retries=3,
    )
    with pytest.raises(LLMRequestError) as captured:
        client.generate("test")
    assert calls == 1
    assert captured.value.status_code == 401
    assert captured.value.attempts == 1
