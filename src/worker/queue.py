"""Durable single-box job queue (SQLite, stdlib only).

A job is "run bench for this config dir". State machine:
pending -> leased -> done|failed. A leased job whose lease expires (the
worker crashed/was killed) is reclaimable by the next claim() — this is
the crash-safety that makes "queue the agents" meaningful. No daemon,
no external deps; WAL + BEGIN IMMEDIATE gives atomic single-box claims.
"""

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_path TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending',
    owner TEXT,
    lease_expires REAL NOT NULL DEFAULT 0,
    verdict TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, id);
"""


class Queue:
    def __init__(self, db_path: str):
        self.path = str(Path(db_path).resolve())
        self._db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def enqueue(self, config_path: str, attempt: int = 0) -> int:
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO jobs(config_path,attempt,state,created,updated) VALUES(?,?,'pending',?,?)",
            (str(Path(config_path).resolve()), attempt, now, now),
        )
        rid = cur.lastrowid
        assert rid is not None
        return int(rid)

    def claim(self, owner: str, lease_secs: float = 600.0) -> dict | None:
        """Atomically take the oldest pending job, or one whose lease
        expired (worker died). Returns the job row as a dict or None."""
        now = time.time()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE state='pending' "
                "OR (state='leased' AND lease_expires < ?) ORDER BY id LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                self._db.execute("COMMIT")
                return None
            self._db.execute(
                "UPDATE jobs SET state='leased',owner=?,lease_expires=?,updated=? WHERE id=?",
                (owner, now + lease_secs, now, row["id"]),
            )
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        return dict(row)

    def heartbeat(self, job_id: int, owner: str, lease_secs: float = 600.0) -> None:
        self._db.execute(
            "UPDATE jobs SET lease_expires=?,updated=? WHERE id=? AND owner=? AND state='leased'",
            (time.time() + lease_secs, time.time(), job_id, owner),
        )

    def complete(self, job_id: int, owner: str, verdict: dict) -> None:
        self._finish(job_id, owner, "done", verdict)

    def fail(self, job_id: int, owner: str, detail: str) -> None:
        self._finish(job_id, owner, "failed", {"error": detail})

    def _finish(self, job_id: int, owner: str, state: str, verdict: dict) -> None:
        self._db.execute(
            "UPDATE jobs SET state=?,verdict=?,updated=? WHERE id=? AND owner=?",
            (state, json.dumps(verdict), time.time(), job_id, owner),
        )

    def get(self, job_id: int) -> dict | None:
        row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict:
        rows = self._db.execute("SELECT state,COUNT(*) c FROM jobs GROUP BY state").fetchall()
        return {r["state"]: r["c"] for r in rows}
