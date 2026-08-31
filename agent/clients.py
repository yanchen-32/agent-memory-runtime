from __future__ import annotations

import json
import os
import re
import time
from urllib import error, request


class LLMRequestError(RuntimeError):
    """Safe, structured terminal error for an LLM HTTP request."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
        self.retryable = retryable


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
        if not best_line:
            return "UNKNOWN"

        question_stem = re.sub(
            r"(?:是什么|什么|哪一个|哪个|多少|哪里|谁|[？?])",
            "",
            question,
        ).replace(" ", "")
        compact_content = best_line.replace(" ", "")
        if question_stem and compact_content.startswith(question_stem):
            consumed = 0
            split_at = 0
            for split_at, character in enumerate(best_line, start=1):
                if not character.isspace():
                    consumed += 1
                if consumed >= len(question_stem):
                    break
            short_answer = best_line[split_at:].strip(" ：:，,。.!！?？")
            short_answer = re.sub(r"^(?:是|为)\s*", "", short_answer)
            if short_answer:
                return short_answer
        return best_line


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
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        thinking: str | None = "disabled",
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ):
        self.model = model or os.getenv("LLM_MODEL", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.last_usage: dict = {}
        self.last_attempts = 0
        if not self.model:
            raise ValueError("LLM model is required. Pass model=... or set LLM_MODEL.")
        if thinking not in {None, "disabled", "enabled"}:
            raise ValueError("thinking must be 'disabled', 'enabled', or None")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")

    def generate(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.thinking is not None:
            body["thinking"] = {"type": self.thinking}
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        self.last_usage = {}
        for attempt in range(1, self.max_retries + 2):
            self.last_attempts = attempt
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.last_usage = data.get("usage", {})
                return data["choices"][0]["message"]["content"].strip()
            except error.HTTPError as exc:
                retryable = exc.code in retryable_statuses
                if not retryable or attempt > self.max_retries:
                    raise LLMRequestError(
                        f"LLM request failed with HTTP {exc.code}",
                        attempts=attempt,
                        status_code=exc.code,
                        retryable=retryable,
                    ) from exc
            except (error.URLError, TimeoutError, OSError) as exc:
                if attempt > self.max_retries:
                    raise LLMRequestError(
                        "LLM request failed with a transient network error",
                        attempts=attempt,
                        retryable=True,
                    ) from exc
            time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))

        raise AssertionError("unreachable retry loop")
