from __future__ import annotations

import json
from typing import Callable

from memory.schema import MemoryCandidate, MemoryType


class RuleMemoryClassifier:
    """Rule-first Working/Episodic/Semantic classifier for V1."""

    working_markers = ("正在", "当前", "这一步", "临时", "稍后", "待会", "调试")
    semantic_predicates = {"喜欢", "偏好", "使用", "部署平台", "数据库", "架构", "项目名称"}
    episodic_markers = ("完成", "提交", "参加", "修改", "改为", "改成", "昨天", "今天", "明天")

    def classify(self, candidate: MemoryCandidate) -> tuple[MemoryType, float]:
        text = candidate.content
        if any(marker in text for marker in self.working_markers):
            return MemoryType.WORKING, 0.80
        if candidate.event_time is not None or any(marker in text for marker in self.episodic_markers):
            return MemoryType.EPISODIC, 0.82
        if candidate.predicate in self.semantic_predicates or candidate.subject is not None:
            return MemoryType.SEMANTIC, 0.84
        return MemoryType.EPISODIC, 0.60


class LLMMemoryClassifier:
    def __init__(self, json_generator: Callable[[str], str | dict]):
        self.json_generator = json_generator

    def classify(self, candidate: MemoryCandidate) -> tuple[MemoryType, float]:
        prompt = (
            "Classify the memory as working, episodic, or semantic. "
            "Return JSON with memory_type and confidence.\nMemory: " + candidate.content
        )
        raw = self.json_generator(prompt)
        if isinstance(raw, str):
            raw = json.loads(raw)
        return MemoryType(raw["memory_type"]), float(raw.get("confidence", 0.8))
