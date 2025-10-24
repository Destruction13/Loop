"""SQLite repository for user data."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from app.models import UserProfile


class Repository:
    """Repository layer encapsulating SQLite operations."""

    def __init__(self, db_path: Path, daily_limit: int) -> None:
        self._db_path = db_path
        self._daily_limit = daily_limit
        self._lock: asyncio.Lock | None = None

    async def init(self) -> None:
        """Initialize database schema."""

        await asyncio.to_thread(self._create_schema)

    async def get_user(self, user_id: int) -> Optional[UserProfile]:
        return await asyncio.to_thread(self._get_user_sync, user_id)

    async def ensure_user(self, user_id: int) -> UserProfile:
        profile = await self.get_user(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, last_reset_at=datetime.now(UTC))
            await asyncio.to_thread(self._upsert_user, profile)
        return profile

    async def update_filters(self, user_id: int, *, gender: Optional[str] = None, age_bucket: Optional[str] = None, style: Optional[str] = None) -> None:
        await asyncio.to_thread(
            self._update_filters_sync, user_id, gender, age_bucket, style
        )

    async def ensure_daily_reset(self, user_id: int, *, now: Optional[datetime] = None) -> UserProfile:
        now = now or datetime.now(UTC)
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            if not profile.last_reset_at or now - profile.last_reset_at >= timedelta(hours=24):
                profile.daily_used = 0
                profile.seen_models = []
                profile.last_reset_at = now
                await asyncio.to_thread(self._upsert_user, profile)
            return profile

    async def remaining_tries(self, user_id: int) -> int:
        profile = await self.ensure_daily_reset(user_id)
        return profile.remaining(self._daily_limit)

    async def inc_used_on_success(self, user_id: int) -> None:
        profile = await self.ensure_daily_reset(user_id)
        lock = self._ensure_lock()
        async with lock:
            profile.daily_used += 1
            await asyncio.to_thread(self._upsert_user, profile)

    async def add_seen_models(self, user_id: int, model_ids: Iterable[str]) -> None:
        profile = await self.ensure_daily_reset(user_id)
        lock = self._ensure_lock()
        async with lock:
            seen_set = set(profile.seen_models)
            seen_set.update(model_ids)
            profile.seen_models = list(seen_set)
            await asyncio.to_thread(self._upsert_user, profile)

    async def set_referrer(self, user_id: int, referrer_id: int) -> None:
        profile = await self.ensure_user(user_id)
        profile.referrer_id = referrer_id
        await asyncio.to_thread(self._upsert_user, profile)

    async def set_reminder(self, user_id: int, when: Optional[datetime]) -> None:
        profile = await self.ensure_user(user_id)
        profile.remind_at = when
        await asyncio.to_thread(self._upsert_user, profile)

    async def list_due_reminders(self, now: datetime) -> List[UserProfile]:
        return await asyncio.to_thread(self._list_due_reminders_sync, now)

    # internal helpers
    def _create_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    gender TEXT,
                    age_bucket TEXT,
                    style TEXT,
                    daily_used INTEGER DEFAULT 0,
                    last_reset_at TEXT,
                    seen_models TEXT,
                    remind_at TEXT,
                    referrer_id INTEGER
                )
                """
            )
            conn.commit()

    def _get_user_sync(self, user_id: int) -> Optional[UserProfile]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        if not row:
            return None
        last_reset_at = (
            datetime.fromisoformat(row["last_reset_at"]) if row["last_reset_at"] else None
        )
        remind_at = datetime.fromisoformat(row["remind_at"]) if row["remind_at"] else None
        seen_models = json.loads(row["seen_models"]) if row["seen_models"] else []
        return UserProfile(
            user_id=row["user_id"],
            gender=row["gender"],
            age_bucket=row["age_bucket"],
            style=row["style"] or "normal",
            daily_used=row["daily_used"],
            last_reset_at=last_reset_at,
            seen_models=seen_models,
            remind_at=remind_at,
            referrer_id=row["referrer_id"],
        )

    def _update_filters_sync(
        self, user_id: int, gender: Optional[str], age_bucket: Optional[str], style: Optional[str]
    ) -> None:
        profile = self._get_user_sync(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
        if gender:
            profile.gender = gender
        if age_bucket:
            profile.age_bucket = age_bucket
        if style:
            profile.style = style
        self._upsert_user(profile)

    def _upsert_user(self, profile: UserProfile) -> None:
        data = asdict(profile)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, gender, age_bucket, style, daily_used, last_reset_at, seen_models, remind_at, referrer_id)
                VALUES (:user_id, :gender, :age_bucket, :style, :daily_used, :last_reset_at, :seen_models, :remind_at, :referrer_id)
                ON CONFLICT(user_id) DO UPDATE SET
                    gender=excluded.gender,
                    age_bucket=excluded.age_bucket,
                    style=excluded.style,
                    daily_used=excluded.daily_used,
                    last_reset_at=excluded.last_reset_at,
                    seen_models=excluded.seen_models,
                    remind_at=excluded.remind_at,
                    referrer_id=excluded.referrer_id
                """,
                {
                    "user_id": data["user_id"],
                    "gender": data["gender"],
                    "age_bucket": data["age_bucket"],
                    "style": data["style"],
                    "daily_used": data["daily_used"],
                    "last_reset_at": data["last_reset_at"].isoformat()
                    if data["last_reset_at"]
                    else None,
                    "seen_models": json.dumps(data["seen_models"]),
                    "remind_at": data["remind_at"].isoformat() if data["remind_at"] else None,
                    "referrer_id": data["referrer_id"],
                },
            )
            conn.commit()

    def _list_due_reminders_sync(self, now: datetime) -> List[UserProfile]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM users WHERE remind_at IS NOT NULL AND remind_at <= ?",
                (now.isoformat(),),
            )
            rows = cur.fetchall()
        result = []
        for row in rows:
            profile = self._get_user_sync(row["user_id"])
            if profile:
                result.append(profile)
        return result

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.close()
