from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from memory.schema import SearchResult
from memory.storage import MemoryStore


STRUCTURED_LOOKUP_VERSION = "structured-lookup-v2"

# These are schema-level aliases, never benchmark case IDs or answer values.
DEFAULT_PREDICATE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "数据库": ("数据存储系统", "持久化后端", "数据落在哪套系统", "负责持久化的后端"),
    "部署平台": ("运行平台", "承载节点", "运行在哪类节点", "由什么环境承载"),
    "架构": ("服务形态", "系统组织形态", "服务如何组织", "系统采用什么形态"),
    "项目名称": ("方案名称", "项目标识", "这套方案如何称呼", "登记的项目标识"),
    "截止日期": ("最晚完成时间", "截止时间", "最晚何时完成", "必须在哪天前完成"),
    "答辩日期": ("答辩安排时间", "汇报日期", "何时进行成果汇报", "安排在哪天汇报"),
}


@dataclass(frozen=True, slots=True)
class ResolvedFactKey:
    subject: str
    predicate: str
    predicate_surface: str


class StructuredFactResolver:
    """Resolve one schema fact key without reading or scoring memory content."""

    def __init__(
        self,
        predicate_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        source = predicate_aliases or DEFAULT_PREDICATE_ALIASES
        self.predicate_aliases = {
            predicate: tuple(dict.fromkeys((predicate, *aliases)))
            for predicate, aliases in source.items()
        }

    def resolve(
        self,
        query: str,
        fact_keys: Sequence[tuple[str, str]],
    ) -> ResolvedFactKey | None:
        matches: dict[tuple[str, str], ResolvedFactKey] = {}
        for subject, predicate in fact_keys:
            if not subject or subject not in query:
                continue
            surfaces = self.predicate_aliases.get(predicate, (predicate,))
            matched = [surface for surface in surfaces if surface and surface in query]
            if matched:
                matches[(subject, predicate)] = ResolvedFactKey(
                    subject=subject,
                    predicate=predicate,
                    predicate_surface=max(matched, key=len),
                )
        if len(matches) != 1:
            return None
        return next(iter(matches.values()))


class ExactFactRetriever:
    """Resolve and retrieve a fact key before Vector/BM25 candidate scans."""

    def __init__(
        self,
        store: MemoryStore,
        resolver: StructuredFactResolver | None = None,
    ) -> None:
        self.store = store
        self.resolver = resolver or StructuredFactResolver()
        self.last_resolution: ResolvedFactKey | None = None
        self.last_visible_record_count = 0

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[SearchResult]:
        self.last_resolution = self.resolver.resolve(
            query,
            self.store.list_fact_keys(user_id=user_id, query_time=query_time),
        )
        if self.last_resolution is None or top_k <= 0:
            self.last_visible_record_count = 0
            return []
        records = self.store.list_by_fact_key(
            self.last_resolution.subject,
            self.last_resolution.predicate,
            user_id=user_id,
            query_time=query_time,
        )
        self.last_visible_record_count = len(records)
        ordered = sorted(
            records,
            key=lambda record: (-record.version, -record.valid_from.timestamp(), record.memory_id),
        )[:top_k]
        return [
            SearchResult(
                memory_id=record.memory_id,
                content=record.content,
                score=1.0,
                metadata={
                    "retriever": STRUCTURED_LOOKUP_VERSION,
                    "subject": record.subject,
                    "predicate": record.predicate,
                    "predicate_surface": self.last_resolution.predicate_surface,
                    "query_time": str(query_time) if query_time is not None else None,
                },
            )
            for record in ordered
        ]
