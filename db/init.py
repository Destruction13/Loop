from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

@dataclass(slots=True)
class SessionInfo:
    tg_id: int
    attempt_count: int
    last_activity_ts: datetime
    ecom_prompt_sent: bool
    social_ad_sent: bool


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def init(self, schema_path: Path) -> None:
        await self._run_in_thread(self._apply_schema, schema_path)

    async def upsert_user(
        self,
        tg_id: int,
        username: str | None,
        name: str | None,
    ) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        await self._execute(
            """
            INSERT INTO users (tg_id, username, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tg_id) DO UPDATE SET
                username=excluded.username,
                name=excluded.name,
                updated_at=excluded.updated_at
            """,
            (tg_id, username, name, now, now),
        )
        await self._ensure_session_exists(tg_id)

    async def update_gender(self, tg_id: int, gender: str) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        await self._execute(
            "UPDATE users SET gender = ?, updated_at = ? WHERE tg_id = ?",
            (gender, now, tg_id),
        )

    async def set_phone(self, tg_id: int, phone: str) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        await self._execute(
            "UPDATE users SET phone = ?, updated_at = ? WHERE tg_id = ?",
            (phone, now, tg_id),
        )

    async def get_user(self, tg_id: int) -> dict[str, Any] | None:
        rows = await self._fetchall(
            "SELECT tg_id, username, name, gender, phone, created_at, updated_at FROM users WHERE tg_id = ?",
            (tg_id,),
        )
        if not rows:
            return None
        return dict(rows[0])

    async def increment_attempt_count(self, tg_id: int) -> int:
        await self._ensure_session_exists(tg_id)
        await self._execute(
            """
            UPDATE sessions
            SET attempt_count = attempt_count + 1,
                last_activity_ts = ?
            WHERE tg_id = ?
            """,
            (self._now(), tg_id),
        )
        info = await self.get_session(tg_id)
        return info.attempt_count if info else 0

    async def reset_attempts(self, tg_id: int) -> None:
        await self._execute(
            "UPDATE sessions SET attempt_count = 0 WHERE tg_id = ?",
            (tg_id,),
        )

    async def log_event(self, tg_id: int | None, event_type: str, payload: dict[str, Any] | None = None) -> None:
        await self._execute(
            "INSERT INTO events (tg_id, type, payload_json, ts) VALUES (?, ?, ?, ?)",
            (
                tg_id,
                event_type,
                json.dumps(payload or {}, ensure_ascii=False),
                self._now(),
            ),
        )

    async def update_last_activity(self, tg_id: int) -> None:
        await self._ensure_session_exists(tg_id)
        await self._execute(
            """
            UPDATE sessions
            SET last_activity_ts = ?,
                ecom_prompt_sent = 0,
                social_ad_sent = 0
            WHERE tg_id = ?
            """,
            (self._now(), tg_id),
        )

    async def mark_ecom_prompt_sent(self, tg_id: int) -> None:
        await self._execute(
            "UPDATE sessions SET ecom_prompt_sent = 1 WHERE tg_id = ?",
            (tg_id,),
        )

    async def mark_social_ad_sent(self, tg_id: int) -> None:
        await self._execute(
            "UPDATE sessions SET social_ad_sent = 1 WHERE tg_id = ?",
            (tg_id,),
        )

    async def get_session(self, tg_id: int) -> SessionInfo | None:
        rows = await self._fetchall(
            "SELECT tg_id, attempt_count, last_activity_ts, ecom_prompt_sent, social_ad_sent FROM sessions WHERE tg_id = ?",
            (tg_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return SessionInfo(
            tg_id=row["tg_id"],
            attempt_count=row["attempt_count"],
            last_activity_ts=datetime.fromisoformat(row["last_activity_ts"].replace("Z", "+00:00")),
            ecom_prompt_sent=bool(row["ecom_prompt_sent"]),
            social_ad_sent=bool(row["social_ad_sent"]),
        )

    async def fetch_users(self) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            "SELECT tg_id, username, name, gender, phone, created_at, updated_at FROM users ORDER BY created_at",
        )
        return [dict(row) for row in rows]

    async def fetch_events_since(self, last_event_id: int | None) -> list[dict[str, Any]]:
        if last_event_id is None:
            query = "SELECT id, tg_id, type, payload_json, ts FROM events ORDER BY id"
            params: Sequence[Any] = ()
        else:
            query = "SELECT id, tg_id, type, payload_json, ts FROM events WHERE id > ? ORDER BY id"
            params = (last_event_id,)
        rows = await self._fetchall(query, params)
        return [dict(row) for row in rows]

    async def _ensure_session_exists(self, tg_id: int) -> None:
        await self._execute(
            """
            INSERT INTO sessions (tg_id, attempt_count, last_activity_ts, ecom_prompt_sent, social_ad_sent)
            VALUES (?, 0, ?, 0, 0)
            ON CONFLICT(tg_id) DO NOTHING
            """,
            (tg_id, self._now()),
        )

    async def _execute(self, query: str, params: Sequence[Any] | None = None) -> None:
        await self._run_in_thread(self._execute_sync, query, params or ())

    async def _fetchall(
        self, query: str, params: Sequence[Any] | None = None
    ) -> list[sqlite3.Row]:
        return await self._run_in_thread(self._fetchall_sync, query, params or ())

    async def _run_in_thread(self, func, *args):
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, func, *args)

    def _apply_schema(self, schema_path: Path) -> None:
        with sqlite3.connect(self._path) as conn, schema_path.open("r", encoding="utf-8") as fh:
            conn.executescript(fh.read())
            conn.commit()

    def _execute_sync(self, query: str, params: Sequence[Any]) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(query, params)
            conn.commit()

    def _fetchall_sync(self, query: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        return rows

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"


__all__ = ["Database", "SessionInfo"]
