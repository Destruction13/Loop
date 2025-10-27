"""SQLite repository for user data."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

from app.models import UserContact, UserProfile


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
            now = datetime.now(timezone.utc)
            profile = UserProfile(
                user_id=user_id,
                last_reset_at=now,
                last_activity_ts=int(time.time()),
            )
            await asyncio.to_thread(self._upsert_user, profile)
        return profile

    async def update_filters(self, user_id: int, *, gender: Optional[str] = None) -> None:
        await asyncio.to_thread(self._update_filters_sync, user_id, gender)

    async def ensure_daily_reset(self, user_id: int, *, now: Optional[datetime] = None) -> UserProfile:
        now = now or datetime.now(timezone.utc)
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

    async def touch_activity(self, user_id: int, *, timestamp: Optional[int] = None) -> None:
        await self.ensure_user(user_id)
        ts = int(timestamp or time.time())
        await asyncio.to_thread(self._update_last_activity_sync, user_id, ts)

    async def list_idle_reminder_candidates(self, threshold_ts: int) -> List[UserProfile]:
        return await asyncio.to_thread(self._list_idle_reminder_candidates_sync, threshold_ts)

    async def mark_idle_reminder_sent(self, user_id: int) -> None:
        await asyncio.to_thread(self._mark_idle_reminder_sent_sync, user_id)

    async def inc_used_on_success(self, user_id: int) -> None:
        profile = await self.ensure_daily_reset(user_id)
        lock = self._ensure_lock()
        async with lock:
            profile.daily_used += 1
            await asyncio.to_thread(self._upsert_user, profile)

    async def get_generation_count(self, user_id: int) -> int:
        profile = await self.ensure_user(user_id)
        return profile.gen_count

    async def increment_generation_count(self, user_id: int) -> int:
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            profile.gen_count += 1
            await asyncio.to_thread(self._upsert_user, profile)
            return profile.gen_count

    async def set_generation_count(self, user_id: int, value: int) -> None:
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            profile.gen_count = max(value, 0)
            await asyncio.to_thread(self._upsert_user, profile)

    async def set_contact_skip_once(self, user_id: int, value: bool) -> None:
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            profile.contact_skip_once = value
            await asyncio.to_thread(self._upsert_user, profile)

    async def set_contact_never(self, user_id: int, value: bool) -> None:
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            profile.contact_never = value
            await asyncio.to_thread(self._upsert_user, profile)

    async def get_user_contact(self, user_id: int) -> Optional[UserContact]:
        return await asyncio.to_thread(self._get_user_contact_sync, user_id)

    async def upsert_user_contact(self, contact: UserContact) -> None:
        await asyncio.to_thread(self._upsert_user_contact_sync, contact)

    async def mark_contact_reward_granted(self, user_id: int) -> None:
        await asyncio.to_thread(self._mark_contact_reward_sync, user_id)

    async def add_seen_models(
        self, user_id: int, model_ids: Iterable[str], *, context: str = "global"
    ) -> None:
        ids = list(dict.fromkeys(model_ids))
        if not ids:
            return
        await self.record_seen_models(user_id, ids, context=context)
        profile = await self.ensure_daily_reset(user_id)
        lock = self._ensure_lock()
        async with lock:
            seen_set = set(profile.seen_models)
            seen_set.update(ids)
            profile.seen_models = list(seen_set)
            await asyncio.to_thread(self._upsert_user, profile)

    async def record_seen_models(
        self,
        user_id: int,
        model_ids: Iterable[str],
        *,
        when: datetime | None = None,
        context: str = "global",
    ) -> None:
        ids = list(dict.fromkeys(model_ids))
        if not ids:
            return
        timestamp = (when or datetime.now(timezone.utc)).isoformat()
        await asyncio.to_thread(
            self._record_seen_models_sync, user_id, ids, timestamp, context
        )

    async def list_seen_models(self, user_id: int, *, context: str) -> set[str]:
        return await asyncio.to_thread(self._list_seen_models_sync, user_id, context)

    async def sync_catalog_version(
        self, version_hash: str, *, clear_on_change: bool
    ) -> Tuple[bool, bool]:
        lock = self._ensure_lock()
        async with lock:
            return await asyncio.to_thread(
                self._sync_catalog_version_sync,
                version_hash,
                clear_on_change,
            )

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
                    referrer_id INTEGER,
                    gen_count INTEGER NOT NULL DEFAULT 0,
                    contact_skip_once INTEGER NOT NULL DEFAULT 0,
                    contact_never INTEGER NOT NULL DEFAULT 0,
                    last_activity_ts INTEGER NOT NULL DEFAULT 0,
                    idle_reminder_sent INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_user_columns(conn)
            self._ensure_seen_table(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version_hash TEXT NOT NULL
                )
                """
            )
            self._ensure_contact_table(conn)
            conn.commit()

    def _row_to_profile(self, row: sqlite3.Row) -> UserProfile:
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
            gen_count=row["gen_count"] if row["gen_count"] is not None else 0,
            contact_skip_once=bool(row["contact_skip_once"] or 0),
            contact_never=bool(row["contact_never"] or 0),
            last_activity_ts=row["last_activity_ts"] if row["last_activity_ts"] else 0,
            idle_reminder_sent=bool(row["idle_reminder_sent"] or 0),
        )

    def _get_user_sync(self, user_id: int) -> Optional[UserProfile]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        if not row:
            return None
        return self._row_to_profile(row)

    def _update_filters_sync(self, user_id: int, gender: Optional[str]) -> None:
        profile = self._get_user_sync(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
        if gender:
            profile.gender = gender
        self._upsert_user(profile)

    def _upsert_user(self, profile: UserProfile) -> None:
        data = asdict(profile)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    gender,
                    age_bucket,
                    style,
                    daily_used,
                    last_reset_at,
                    seen_models,
                    remind_at,
                    referrer_id,
                    gen_count,
                    contact_skip_once,
                    contact_never,
                    last_activity_ts,
                    idle_reminder_sent
                )
                VALUES (
                    :user_id,
                    :gender,
                    :age_bucket,
                    :style,
                    :daily_used,
                    :last_reset_at,
                    :seen_models,
                    :remind_at,
                    :referrer_id,
                    :gen_count,
                    :contact_skip_once,
                    :contact_never,
                    :last_activity_ts,
                    :idle_reminder_sent
                )
                ON CONFLICT(user_id) DO UPDATE SET
                    gender=excluded.gender,
                    age_bucket=excluded.age_bucket,
                    style=excluded.style,
                    daily_used=excluded.daily_used,
                    last_reset_at=excluded.last_reset_at,
                    seen_models=excluded.seen_models,
                    remind_at=excluded.remind_at,
                    referrer_id=excluded.referrer_id,
                    gen_count=excluded.gen_count,
                    contact_skip_once=excluded.contact_skip_once,
                    contact_never=excluded.contact_never,
                    last_activity_ts=excluded.last_activity_ts,
                    idle_reminder_sent=excluded.idle_reminder_sent
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
                    "gen_count": data["gen_count"],
                    "contact_skip_once": 1 if data["contact_skip_once"] else 0,
                    "contact_never": 1 if data["contact_never"] else 0,
                    "last_activity_ts": data["last_activity_ts"],
                    "idle_reminder_sent": 1 if data["idle_reminder_sent"] else 0,
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
        return [self._row_to_profile(row) for row in rows]

    def _list_idle_reminder_candidates_sync(self, threshold_ts: int) -> List[UserProfile]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT * FROM users
                WHERE last_activity_ts > 0
                  AND last_activity_ts <= ?
                  AND idle_reminder_sent = 0
                """,
                (threshold_ts,),
            )
            rows = cur.fetchall()
        return [self._row_to_profile(row) for row in rows]

    def _update_last_activity_sync(self, user_id: int, timestamp: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE users SET last_activity_ts = ?, idle_reminder_sent = 0 WHERE user_id = ?",
                (timestamp, user_id),
            )
            conn.commit()

    def _mark_idle_reminder_sent_sync(self, user_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE users SET idle_reminder_sent = 1 WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()

    def _record_seen_models_sync(
        self, user_id: int, model_ids: list[str], timestamp: str, context: str
    ) -> None:
        if not model_ids:
            return
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO user_seen_models (user_id, context, model_id, seen_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, context, model_id) DO UPDATE SET seen_at=excluded.seen_at
                """,
                [(user_id, context, model_id, timestamp) for model_id in model_ids],
            )
            conn.commit()

    def _list_seen_models_sync(self, user_id: int, context: str) -> set[str]:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT model_id FROM user_seen_models WHERE user_id = ? AND context = ?",
                (user_id, context),
            )
            rows = cur.fetchall()
        return {row[0] for row in rows}

    def _get_user_contact_sync(self, user_id: int) -> Optional[UserContact]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT tg_user_id, phone_e164, source, consent, consent_ts, reward_granted
                FROM user_contacts
                WHERE tg_user_id = ?
                """,
                (user_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return UserContact(
            tg_user_id=row["tg_user_id"],
            phone_e164=row["phone_e164"],
            source=row["source"],
            consent=bool(row["consent"]),
            consent_ts=row["consent_ts"],
            reward_granted=bool(row["reward_granted"]),
        )

    def _upsert_user_contact_sync(self, contact: UserContact) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO user_contacts (
                    tg_user_id,
                    phone_e164,
                    source,
                    consent,
                    consent_ts,
                    reward_granted
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tg_user_id) DO UPDATE SET
                    phone_e164=excluded.phone_e164,
                    source=excluded.source,
                    consent=excluded.consent,
                    consent_ts=excluded.consent_ts,
                    reward_granted=excluded.reward_granted
                """,
                (
                    contact.tg_user_id,
                    contact.phone_e164,
                    contact.source,
                    1 if contact.consent else 0,
                    contact.consent_ts,
                    1 if contact.reward_granted else 0,
                ),
            )
            conn.commit()

    def _mark_contact_reward_sync(self, user_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE user_contacts SET reward_granted = 1 WHERE tg_user_id = ?",
                (user_id,),
            )
            conn.commit()

    def _sync_catalog_version_sync(
        self, version_hash: str, clear_on_change: bool
    ) -> Tuple[bool, bool]:
        changed = False
        cleared = False
        with self._connection() as conn:
            cur = conn.execute("SELECT version_hash FROM catalog_meta WHERE id = 1")
            row = cur.fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO catalog_meta (id, version_hash) VALUES (1, ?)",
                    (version_hash,),
                )
                conn.commit()
                return changed, cleared
            current = row[0]
            if current != version_hash:
                changed = True
                if clear_on_change:
                    conn.execute("DELETE FROM user_seen_models")
                    cleared = True
                conn.execute(
                    "UPDATE catalog_meta SET version_hash = ? WHERE id = 1",
                    (version_hash,),
                )
            conn.commit()
        return changed, cleared

    def _ensure_seen_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_seen_models (
                user_id INTEGER NOT NULL,
                context TEXT NOT NULL DEFAULT 'global',
                model_id TEXT NOT NULL,
                seen_at TEXT NOT NULL,
                PRIMARY KEY (user_id, context, model_id)
            )
            """,
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_seen_models)").fetchall()
        }
        if "context" not in columns:
            conn.execute("ALTER TABLE user_seen_models RENAME TO user_seen_models_legacy")
            conn.execute(
                """
                CREATE TABLE user_seen_models (
                    user_id INTEGER NOT NULL,
                    context TEXT NOT NULL DEFAULT 'global',
                    model_id TEXT NOT NULL,
                    seen_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, context, model_id)
                )
                """,
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO user_seen_models (user_id, context, model_id, seen_at)
                SELECT user_id, 'global', model_id, seen_at FROM user_seen_models_legacy
                """,
            )
            conn.execute("DROP TABLE user_seen_models_legacy")

    def _ensure_user_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "gen_count" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN gen_count INTEGER NOT NULL DEFAULT 0"
            )
        if "contact_skip_once" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN contact_skip_once INTEGER NOT NULL DEFAULT 0"
            )
        if "contact_never" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN contact_never INTEGER NOT NULL DEFAULT 0"
            )
        if "last_activity_ts" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN last_activity_ts INTEGER NOT NULL DEFAULT 0"
            )
        if "idle_reminder_sent" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN idle_reminder_sent INTEGER NOT NULL DEFAULT 0"
            )

    def _ensure_contact_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_contacts (
                tg_user_id INTEGER PRIMARY KEY,
                phone_e164 TEXT NOT NULL,
                source TEXT NOT NULL,
                consent INTEGER NOT NULL,
                consent_ts INTEGER NOT NULL,
                reward_granted INTEGER NOT NULL DEFAULT 0
            )
            """
        )

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
