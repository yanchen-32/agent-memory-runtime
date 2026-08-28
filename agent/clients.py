from __future__ import annotations

import json
import os
import re
from urllib import request


class RuleBasedClient:
    """Offline deterministic client used only to smoke-test the pipeline."""

    def generate(self, prompt: str) -> str:
        memory_lines = [line.strip() for line in prompt.splitlines() if line.startswith("MEMORY[")]
        question_match = re.search(r"QUESTION:\s*(.+)", prompt)
        question = question_match.group(1).strip() if question_match else ""

        if not memory_lines:
            return "UNKNOWN"

        q_tokens = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", question.lower()))
        best_line = ""
        best_score = -1
        for line in memory_lines:
            content = line.split("]", 1)[-1].strip()
            tokens = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", content.lower()))
            score = len(q_tokens & tokens)
            if score > best_score:
                best_score = score
                best_line = content
        return best_line or "UNKNOWN"


class OpenAICompatibleClient:
    """Minimal client for OpenAI-compatible chat-completions endpoints.

    Environment defaults:
      LLM_BASE_URL=http://localhost:8000/v1
      LLM_API_KEY=EMPTY
      LLM_MODEL=<required unless passed to constructor>
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        self.model = model or os.getenv("LLM_MODEL", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.timeout = timeout
        if not self.model:
            raise ValueError("LLM model is required. Pass model=... or set LLM_MODEL.")

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
