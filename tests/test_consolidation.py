from memory import InMemoryMemoryStore, MemoryRecord, MemoryType
from memory.consolidation import MemoryConsolidator


def _episodic(memory_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        memory_type=MemoryType.EPISODIC,
        content=content,
        subject="项目数据库",
        predicate="使用",
        object_value="openGauss",
        importance=0.8,
        confidence=0.9,
    )


def test_consolidation_creates_traceable_semantic_memory_and_is_idempotent():
    store = InMemoryMemoryStore()
    store.add(_episodic("episode-1", "项目数据库使用 openGauss。"))
    store.add(_episodic("episode-2", "项目数据库后来仍使用 openGauss。"))
    engine = MemoryConsolidator(store)

    first = engine.consolidate(user_id="u1")
    second = engine.consolidate(user_id="u1")
    semantic = [record for record in store.list_all("u1") if record.memory_type == MemoryType.SEMANTIC]

    assert len(first.created_ids) == 1
    assert second.created_ids == []
    assert len(second.updated_ids) == 1
    assert len(semantic) == 1
    assert semantic[0].source_ids == ["episode-1", "episode-2"]
    assert semantic[0].metadata["consolidation_engine"] == "episodic-to-semantic-v1"
    assert first.fidelity == 1.0
    assert all(record.status.value == "active" for record in store.list_all("u1"))


def test_consolidation_does_not_merge_conflicting_object_values():
    store = InMemoryMemoryStore()
    store.add(_episodic("episode-old", "项目数据库使用 SQLite。"))
    store.get("episode-old").object_value = "SQLite"
    store.add(_episodic("episode-new", "项目数据库使用 openGauss。"))
    report = MemoryConsolidator(store).consolidate(user_id="u1")

    assert report.created_ids == []
    assert report.groups == []
    assert report.skipped_groups == 2
