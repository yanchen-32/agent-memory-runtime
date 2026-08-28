from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memory.schema import MemoryRecord
from .base import MemoryStore


class SQLiteMemoryStore(MemoryStore):
    """Portable V1 metadata store. JSON payload keeps schema migration simple."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_status ON memories(user_id, status)")
        self.conn.commit()

    def add(self, record: MemoryRecord) -> str:
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        self.conn.execute(
            "INSERT INTO memories(memory_id, user_id, status, created_at, payload) VALUES (?, ?, ?, ?, ?)",
            (record.memory_id, record.user_id, record.status.value, record.created_at.isoformat(), payload),
        )
        self.conn.commit()
        return record.memory_id

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute("SELECT payload FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return MemoryRecord.from_dict(json.loads(row[0])) if row else None

    def update(self, record: MemoryRecord) -> None:
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        cursor = self.conn.execute(
            "UPDATE memories SET user_id = ?, status = ?, created_at = ?, payload = ? WHERE memory_id = ?",
            (record.user_id, record.status.value, record.created_at.isoformat(), payload, record.memory_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(record.memory_id)
        self.conn.commit()

    def list_all(self, user_id: str | None = None) -> list[MemoryRecord]:
        if user_id is None:
            rows = self.conn.execute("SELECT payload FROM memories ORDER BY created_at, memory_id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload FROM memories WHERE user_id = ? ORDER BY created_at, memory_id", (user_id,)
            ).fetchall()
        return [MemoryRecord.from_dict(json.loads(row[0])) for row in rows]

    def close(self) -> None:
        self.conn.close()
