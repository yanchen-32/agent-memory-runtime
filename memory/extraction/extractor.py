from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable, Iterable

from memory.schema import MemoryCandidate


class RuleMemoryExtractor:
    """Deterministic V1 extractor used by tests and offline development.

    It extracts user/tool statements and recognizes common fact forms. The
    production path can swap this implementation for LLMMemoryExtractor.
    """

    _fact_patterns = [
        re.compile(r"(?P<subject>[^。！？]{1,40}?)(?:的)?(?P<predicate>截止日期|答辩日期|截止时间|部署平台|数据库|架构|项目名称)(?:是|为|改为|改成|确定为|：|:)?(?P<object>[^。！？]+)"),
        re.compile(r"(?P<subject>我)(?P<predicate>喜欢|偏好|使用)(?P<object>[^。！？]+)"),
    ]

    def __init__(self, min_chars: int = 4):
        self.min_chars = min_chars

    @staticmethod
    def _normalize_messages(messages: Iterable[dict] | str) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return list(messages)

    @classmethod
    def _extract_fact(cls, content: str) -> tuple[str | None, str | None, str | None]:
        for pattern in cls._fact_patterns:
            match = pattern.search(content)
            if match:
                return (
                    match.group("subject").strip(" ，,：:"),
                    match.group("predicate").strip(),
                    match.group("object").strip(" ，,：:"),
                )
        return None, None, None

    @staticmethod
    def _entities(content: str, subject: str | None) -> list[str]:
        entities: list[str] = []
        if subject and subject != "我":
            entities.append(subject)
        entities.extend(re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", content))
        return list(dict.fromkeys(entities))

    def extract(self, messages: Iterable[dict] | str) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for message in self._normalize_messages(messages):
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if role not in {"user", "tool"} or len(content) < self.min_chars:
                continue
            subject, predicate, object_value = self._extract_fact(content)
            candidates.append(
                MemoryCandidate(
                    content=content,
                    entities=self._entities(content, subject),
                    confidence=0.75 if subject else 0.6,
                    subject=subject,
                    predicate=predicate,
                    object_value=object_value,
                    metadata={"source_role": role, "extractor": "rule-v1"},
                )
            )
        return candidates


class LLMMemoryExtractor:
    """Structured-output V1 extractor.

    json_generator receives one prompt string and may return JSON text, a dict,
    or a list. This keeps the runtime independent of any specific LLM SDK.
    """

    def __init__(self, json_generator: Callable[[str], str | dict | list]):
        self.json_generator = json_generator

    def extract(self, messages: Iterable[dict] | str) -> list[MemoryCandidate]:
        if isinstance(messages, str):
            conversation = messages
        else:
            conversation = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )
        prompt = (
            "Extract reusable memories from the conversation. Return a JSON array. "
            "Each item must contain content, entities, keywords, event_time, confidence, "
            "subject, predicate, object_value. Do not invent facts.\n\n" + conversation
        )
        raw = self.json_generator(prompt)
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            raw = raw.get("memories", [raw])
        candidates: list[MemoryCandidate] = []
        for item in raw:
            event_time = item.get("event_time")
            if isinstance(event_time, str) and event_time:
                try:
                    event_time = datetime.fromisoformat(event_time)
                except ValueError:
                    event_time = None
            candidates.append(
                MemoryCandidate(
                    content=str(item["content"]).strip(),
                    entities=list(item.get("entities") or []),
                    keywords=list(item.get("keywords") or []),
                    event_time=event_time,
                    confidence=float(item.get("confidence", 0.8)),
                    subject=item.get("subject"),
                    predicate=item.get("predicate"),
                    object_value=item.get("object_value"),
                    metadata={"extractor": "llm-v1"},
                )
            )
        return candidates
