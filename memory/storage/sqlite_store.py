from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from memory.schema import MemoryRecord, coerce_datetime
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
                subject TEXT,
                predicate TEXT,
                valid_from TEXT,
                valid_to TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        self._migrate_structured_columns()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user_status "
            "ON memories(user_id, status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_fact_key "
            "ON memories(user_id, subject, predicate, status, valid_from, valid_to)"
        )
        self.conn.commit()

    def _migrate_structured_columns(self) -> None:
        """Add/backfill indexed fact columns for databases created before v2."""
        columns = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(memories)")
        }
        for name in ("subject", "predicate", "valid_from", "valid_to"):
            if name not in columns:
                self.conn.execute(f"ALTER TABLE memories ADD COLUMN {name} TEXT")
        rows = self.conn.execute(
            "SELECT memory_id, payload FROM memories WHERE valid_from IS NULL"
        ).fetchall()
        for memory_id, payload in rows:
            record = MemoryRecord.from_dict(json.loads(payload))
            self.conn.execute(
                "UPDATE memories SET subject = ?, predicate = ?, valid_from = ?, "
                "valid_to = ? WHERE memory_id = ?",
                (
                    record.subject,
                    record.predicate,
                    record.valid_from.isoformat(),
                    record.valid_to.isoformat() if record.valid_to is not None else None,
                    memory_id,
                ),
            )

    def add(self, record: MemoryRecord) -> str:
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        self.conn.execute(
            "INSERT INTO memories(memory_id, user_id, status, created_at, subject, "
            "predicate, valid_from, valid_to, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.memory_id,
                record.user_id,
                record.status.value,
                record.created_at.isoformat(),
                record.subject,
                record.predicate,
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to is not None else None,
                payload,
            ),
        )
        self.conn.commit()
        return record.memory_id

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute("SELECT payload FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return MemoryRecord.from_dict(json.loads(row[0])) if row else None

    def update(self, record: MemoryRecord) -> None:
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        cursor = self.conn.execute(
            "UPDATE memories SET user_id = ?, status = ?, created_at = ?, subject = ?, "
            "predicate = ?, valid_from = ?, valid_to = ?, payload = ? WHERE memory_id = ?",
            (
                record.user_id,
                record.status.value,
                record.created_at.isoformat(),
                record.subject,
                record.predicate,
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to is not None else None,
                payload,
                record.memory_id,
            ),
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

    @staticmethod
    def _visibility_sql(
        query_time: datetime | str | None,
    ) -> tuple[str, list[str]]:
        if query_time is None:
            return "status = 'active'", []
        point = coerce_datetime(query_time)
        if point is None:
            raise ValueError("query_time is required")
        value = point.isoformat()
        return (
            "status != 'archived' "
            "AND julianday(valid_from) <= julianday(?) "
            "AND (valid_to IS NULL OR julianday(?) < julianday(valid_to))",
            [value, value],
        )

    def list_fact_keys(
        self,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[tuple[str, str]]:
        visibility, parameters = self._visibility_sql(query_time)
        clauses = ["subject IS NOT NULL", "predicate IS NOT NULL", visibility]
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        rows = self.conn.execute(
            "SELECT DISTINCT subject, predicate FROM memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY subject, predicate",
            parameters,
        ).fetchall()
        return [(str(subject), str(predicate)) for subject, predicate in rows]

    def list_by_fact_key(
        self,
        subject: str,
        predicate: str,
        user_id: str | None = None,
        query_time: datetime | str | None = None,
    ) -> list[MemoryRecord]:
        visibility, visibility_parameters = self._visibility_sql(query_time)
        clauses = ["subject = ?", "predicate = ?", visibility]
        parameters = [subject, predicate, *visibility_parameters]
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        rows = self.conn.execute(
            "SELECT payload FROM memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY valid_from, memory_id",
            parameters,
        ).fetchall()
        return [MemoryRecord.from_dict(json.loads(row[0])) for row in rows]

    def close(self) -> None:
        self.conn.close()
