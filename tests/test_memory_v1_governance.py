from memory import (
    ConflictDetector,
    ConflictType,
    DedupDecision,
    Deduplicator,
    HashEmbeddingModel,
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryStatus,
    VersionedMemoryUpdater,
)


def test_dedup_and_conflict_and_version_update():
    store = InMemoryMemoryStore()
    old = MemoryRecord(
        memory_id="old",
        content="项目截止日期是9月10日。",
        subject="项目",
        predicate="截止日期",
        object_value="9月10日",
    )
    store.add(old)
    dedup = Deduplicator(HashEmbeddingModel(dim=256), similarity_threshold=0.80)
    same = MemoryRecord(content="项目截止日期是9月10日。", subject="项目", predicate="截止日期", object_value="9月10日")
    assert dedup.check(same, store.list_active()).decision == DedupDecision.EXACT_DUPLICATE

    new = MemoryRecord(content="项目截止日期改为9月15日。", subject="项目", predicate="截止日期", object_value="9月15日")
    conflict = ConflictDetector().detect(new, store.list_active())
    assert conflict.conflict_type == ConflictType.VALUE_CONFLICT
    updated = VersionedMemoryUpdater(store).supersede(old, new)
    assert store.get("old").status == MemoryStatus.SUPERSEDED
    assert updated.version == 2
    assert updated.version_group == old.version_group
    assert store.list_active()[0].object_value == "9月15日"
