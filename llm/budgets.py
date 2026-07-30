"""Persistent per-provider usage ledger (the ``provider_usage`` table).

Tracks daily request/token counts and cooldowns so the router can *proactively*
skip exhausted or cooling providers instead of burning a request to discover a
429. Usage is keyed by UTC date, so it resets automatically at midnight — a new
day is simply a new row with zero counts and no cooldown.

Pure storage + clock arithmetic (no LLM calls); the ``now_fn`` seam makes the
time-dependent behavior deterministic under test.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_usage (
  provider       TEXT NOT NULL,
  usage_date     DATE NOT NULL,          -- UTC
  requests       INTEGER DEFAULT 0,
  tokens         INTEGER DEFAULT 0,
  cooldown_until TIMESTAMP,
  PRIMARY KEY (provider, usage_date)
);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_PG_SCHEMA = _SCHEMA.replace("DATE NOT NULL", "DATE NOT NULL").replace(
    "TIMESTAMP", "TIMESTAMPTZ"
)


class BudgetLedger:
    """Per-provider daily usage. Backed by SQLite or Postgres.

    Postgres matters more than it looks: in a container the SQLite file is
    ephemeral, so every restart would hand the router a fresh, empty budget and
    it would cheerfully re-spend a free tier it had already exhausted.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._now = now_fn
        self._lock = threading.Lock()
        self._pg = db_path.startswith("postgresql://") or db_path.startswith("postgres://")
        if self._pg:
            from memory_server.storage.postgres import connect

            self._conn = connect(db_path)
            with self._lock:
                self._conn.execute(_PG_SCHEMA)
            return
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- dialect ------------------------------------------------------------
    def _sql(self, statement: str) -> str:
        """SQLite is the source of truth; Postgres gets the two rewrites it needs."""
        if not self._pg:
            return statement
        return statement.replace("?", "%s").replace("INSERT OR IGNORE", "INSERT")

    def _exec(self, statement: str, params: tuple = ()):
        sql = self._sql(statement)
        if self._pg and "INSERT INTO provider_usage" in sql:
            sql += " ON CONFLICT (provider, usage_date) DO NOTHING"
        cur = self._conn.execute(sql, params)
        if not self._pg:
            self._conn.commit()
        return cur

    # -- helpers -----------------------------------------------------------
    def _now_utc(self) -> datetime:
        return self._now().astimezone(timezone.utc)

    def _today(self) -> str:
        return self._now_utc().date().isoformat()

    def _ensure_row(self, provider: str) -> None:
        self._exec(
            "INSERT OR IGNORE INTO provider_usage (provider, usage_date) VALUES (?, ?)",
            (provider, self._today()),
        )

    # -- writes ------------------------------------------------------------
    def record_request(self, provider: str, tokens: int = 0) -> None:
        with self._lock:
            self._ensure_row(provider)
            self._exec(
                "UPDATE provider_usage SET requests = requests + 1, tokens = tokens + ? "
                "WHERE provider = ? AND usage_date = ?",
                (int(tokens), provider, self._today()),
            )

    def set_cooldown(self, provider: str, until: datetime) -> None:
        with self._lock:
            self._ensure_row(provider)
            self._exec(
                "UPDATE provider_usage SET cooldown_until = ? "
                "WHERE provider = ? AND usage_date = ?",
                (until.astimezone(timezone.utc).isoformat(), provider, self._today()),
            )

    def cooldown_for_seconds(self, provider: str, seconds: float) -> None:
        self.set_cooldown(provider, self._now_utc() + timedelta(seconds=seconds))

    def cooldown_until_utc_midnight(self, provider: str) -> None:
        now = self._now_utc()
        next_midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(
            days=1
        )
        self.set_cooldown(provider, next_midnight)

    # -- reads -------------------------------------------------------------
    def get_usage(self, provider: str) -> tuple[int, int]:
        row = self._exec(
            "SELECT requests, tokens FROM provider_usage WHERE provider = ? AND usage_date = ?",
            (provider, self._today()),
        ).fetchone()
        return (int(row["requests"]), int(row["tokens"])) if row else (0, 0)

    def cooldown_remaining(self, provider: str) -> float:
        row = self._exec(
            "SELECT cooldown_until FROM provider_usage WHERE provider = ? AND usage_date = ?",
            (provider, self._today()),
        ).fetchone()
        if not row or not row["cooldown_until"]:
            return 0.0
        stored = row["cooldown_until"]
        # Postgres hands back a datetime; SQLite an ISO string.
        until = stored if isinstance(stored, datetime) else datetime.fromisoformat(stored)
        return max(0.0, (until - self._now_utc()).total_seconds())

    def is_cooling(self, provider: str) -> bool:
        return self.cooldown_remaining(provider) > 0.0

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Today's usage per provider — handy for the UI/debug panel."""
        rows = self._exec(
            "SELECT provider, requests, tokens FROM provider_usage WHERE usage_date = ?",
            (self._today(),),
        ).fetchall()
        return {
            r["provider"]: {
                "requests": int(r["requests"]),
                "tokens": int(r["tokens"]),
                "cooldown_remaining": self.cooldown_remaining(r["provider"]),
            }
            for r in rows
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
