from datetime import datetime, timezone
import json
import sqlite3

from memory import (
    ExactFactRetriever,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    SQLiteMemoryStore,
)


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


def test_sqlite_exact_spo_lookup_uses_index_and_honors_validity(tmp_path, monkeypatch):
    store = SQLiteMemoryStore(tmp_path / "structured.db")
    old = MemoryRecord(
        memory_id="old",
        user_id="u1",
        content="系统数据库为 SQLite。",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 2, 1, tzinfo=timezone.utc),
        status=MemoryStatus.SUPERSEDED,
        subject="系统",
        predicate="数据库",
        object_value="SQLite",
    )
    current = MemoryRecord(
        memory_id="current",
        user_id="u1",
        content="系统数据库改为 GaussDB。",
        created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 2, 1, tzinfo=timezone.utc),
        subject="系统",
        predicate="数据库",
        object_value="GaussDB",
        version=2,
    )
    store.add(old)
    store.add(current)
    monkeypatch.setattr(
        store,
        "list_all",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full scan")),
    )
    retriever = ExactFactRetriever(store)
    hits = retriever.search(
        "系统目前采用哪一种持久化后端？",
        user_id="u1",
        query_time="2026-03-01T00:00:00+00:00",
    )
    assert [hit.memory_id for hit in hits] == ["current"]
    historical = retriever.search(
        "系统当时的数据存储系统是什么？",
        user_id="u1",
        query_time="2026-01-15T00:00:00+00:00",
    )
    assert [hit.memory_id for hit in historical] == ["old"]
    store.close()


def test_sqlite_migrates_legacy_payload_into_structured_index(tmp_path):
    path = tmp_path / "legacy.db"
    record = MemoryRecord(
        memory_id="legacy",
        user_id="u1",
        content="系统架构为 微服务。",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        subject="系统",
        predicate="架构",
        object_value="微服务",
    )
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
        "status TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
        (
            record.memory_id,
            record.user_id,
            record.status.value,
            record.created_at.isoformat(),
            json.dumps(record.to_dict(), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteMemoryStore(path)
    assert store.list_fact_keys(user_id="u1") == [("系统", "架构")]
    assert [item.memory_id for item in store.list_by_fact_key("系统", "架构", "u1")] == [
        "legacy"
    ]
    store.close()
