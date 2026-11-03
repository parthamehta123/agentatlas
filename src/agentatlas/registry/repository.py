"""SQLite-backed canonical registry (swap for Postgres + pgvector in production)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from agentatlas.registry.models import AgentRecord, Kind, Status

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    agent_id TEXT PRIMARY KEY,
    kind     TEXT NOT NULL,
    status   INTEGER NOT NULL,
    doc      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_entities_kind_status ON entities(kind, status);
"""


class Registry:
    def __init__(self, path: Path | str = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def upsert_many(self, records: Iterable[AgentRecord]) -> int:
        rows = [(r.agent_id, r.kind.value, int(r.status), r.model_dump_json())
                for r in records]
        with self.conn:
            self.conn.executemany(
                "INSERT INTO entities VALUES (?,?,?,?) ON CONFLICT(agent_id) DO UPDATE SET "
                "kind=excluded.kind, status=excluded.status, doc=excluded.doc", rows)
        return len(rows)

    def get(self, agent_id: str) -> AgentRecord | None:
        row = self.conn.execute("SELECT doc FROM entities WHERE agent_id=?",
                                (agent_id,)).fetchone()
        return AgentRecord.model_validate_json(row[0]) if row else None

    def agents(self, min_status: Status = Status.DISCOVERED) -> list[AgentRecord]:
        rows = self.conn.execute(
            "SELECT doc FROM entities WHERE kind=? AND status>=? ORDER BY agent_id",
            (Kind.AGENT.value, int(min_status))).fetchall()
        return [AgentRecord.model_validate_json(r[0]) for r in rows]

    def counts(self) -> dict[str, int]:
        out = {}
        for kind, status, n in self.conn.execute(
                "SELECT kind, status, COUNT(*) FROM entities GROUP BY kind, status"):
            label = Status(status).name if kind == Kind.AGENT.value else kind
            out[label] = out.get(label, 0) + n
        return out
