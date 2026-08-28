from memory import MemoryRecord, MemoryStatus, MemoryType, SQLiteMemoryStore


def test_sqlite_storage_roundtrip_and_status(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    record = MemoryRecord(
        memory_id="m1",
        user_id="u1",
        memory_type=MemoryType.SEMANTIC,
        content="项目数据库为 openGauss。",
        entities=["openGauss"],
        subject="项目",
        predicate="数据库",
        object_value="openGauss",
    )
    store.add(record)
    loaded = store.get("m1")
    assert loaded is not None
    assert loaded.content == record.content
    assert loaded.memory_type == MemoryType.SEMANTIC
    loaded.status = MemoryStatus.ARCHIVED
    store.update(loaded)
    assert store.get("m1").status == MemoryStatus.ARCHIVED
    assert store.list_active("u1") == []
    store.close()
