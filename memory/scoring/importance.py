from __future__ import annotations

from dataclasses import dataclass

from memory.schema import MemoryCandidate


@dataclass(slots=True)
class ImportanceBreakdown:
    score: float
    long_term_value: float
    task_relevance: float
    user_emphasis: float
    future_reuse: float


class ImportanceScorer:
    """Transparent multi-feature V1 importance score in [0, 1]."""

    long_term_terms = ("长期", "最终", "确定", "固定", "项目名称", "部署平台", "数据库", "截止", "答辩")
    task_terms = ("项目", "任务", "实验", "比赛", "论文", "申报", "开发", "benchmark", "agent")
    emphasis_terms = ("记住", "务必", "重要", "必须", "以后", "最终确定")
    reuse_terms = ("偏好", "喜欢", "使用", "名称", "架构", "平台", "数据库", "截止日期")

    @staticmethod
    def _feature(text: str, terms: tuple[str, ...]) -> float:
        hits = sum(1 for term in terms if term.lower() in text.lower())
        return min(1.0, hits / 2.0)

    def breakdown(self, candidate: MemoryCandidate) -> ImportanceBreakdown:
        text = candidate.content
        l = self._feature(text, self.long_term_terms)
        t = self._feature(text, self.task_terms)
        e = self._feature(text, self.emphasis_terms)
        f = self._feature(text, self.reuse_terms)
        if candidate.subject and candidate.predicate:
            l = max(l, 0.65)
            f = max(f, 0.65)
        score = 0.35 * l + 0.30 * t + 0.20 * e + 0.15 * f
        score = min(1.0, max(0.05, score))
        return ImportanceBreakdown(score, l, t, e, f)

    def score(self, candidate: MemoryCandidate) -> float:
        return self.breakdown(candidate).score
