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
from typing import Any, Generator, Iterable, List, Optional, Tuple

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
                daily_used=0,
                tries_used=0,
                daily_try_limit=self._daily_limit,
                cycle_index=0,
                cycle_started_at=None,
                locked_until=None,
                nudge_sent_cycle=False,
                last_reset_at=now,
                last_activity_ts=int(time.time()),
            )
            await asyncio.to_thread(self._upsert_user, profile)
            return profile

        changed = self._apply_profile_defaults(profile)
        if changed:
            await asyncio.to_thread(self._upsert_user, profile)
        return profile

    async def get_current_cycle(self, user_id: int) -> int:
        profile = await self.ensure_user(user_id)
        return profile.cycle_index or 0

    async def start_new_tryon_cycle(self, user_id: int) -> int:
        """Advance the per-user try-on cycle index."""

        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            profile.cycle_index = (profile.cycle_index or 0) + 1
            await asyncio.to_thread(self._upsert_user, profile)
            return profile.cycle_index

    async def update_filters(self, user_id: int, *, gender: Optional[str] = None) -> None:
        await asyncio.to_thread(self._update_filters_sync, user_id, gender)

    async def ensure_daily_reset(
        self,
        user_id: int,
        *,
        now: Optional[datetime] = None,
        lock: bool = True,
    ) -> UserProfile:
        moment = now or datetime.now(timezone.utc)
        if lock:
            lock_obj = self._ensure_lock()
            async with lock_obj:
                return await self._ensure_daily_reset_locked(user_id, moment)
        return await self._ensure_daily_reset_locked(user_id, moment)

    async def _ensure_daily_reset_locked(
        self, user_id: int, moment: datetime
    ) -> UserProfile:
        profile = await self.ensure_user(user_id)
        updated = self._apply_profile_defaults(profile)
        if profile.locked_until and profile.locked_until <= moment:
            self._start_new_cycle(profile, moment)
            updated = True
        if updated:
            await asyncio.to_thread(self._upsert_user, profile)
        return profile

    async def remaining_tries(self, user_id: int) -> int:
        profile = await self.ensure_daily_reset(user_id)
        effective_limit = profile.daily_try_limit or self._daily_limit
        return profile.remaining(effective_limit)

    async def touch_activity(self, user_id: int, *, timestamp: Optional[int] = None) -> None:
        await self.ensure_user(user_id)
        ts = int(timestamp or time.time())
        await asyncio.to_thread(self._update_last_activity_sync, user_id, ts)

    async def list_idle_reminder_candidates(self, threshold_ts: int) -> List[UserProfile]:
        return await asyncio.to_thread(self._list_idle_reminder_candidates_sync, threshold_ts)

    async def mark_idle_reminder_sent(self, user_id: int) -> None:
        await asyncio.to_thread(self._mark_idle_reminder_sent_sync, user_id)

    async def mark_cycle_nudge_sent(self, user_id: int) -> None:
        await asyncio.to_thread(self._set_nudge_flag_sync, user_id, True)

    async def list_social_ad_candidates(self, threshold_ts: int) -> List[UserProfile]:
        return await asyncio.to_thread(self._list_social_ad_candidates_sync, threshold_ts)

    async def mark_social_ad_shown(self, user_id: int) -> None:
        await asyncio.to_thread(self._mark_social_ad_shown_sync, user_id)

    async def inc_used_on_success(
        self, user_id: int, *, now: Optional[datetime] = None
    ) -> None:
        moment = now or datetime.now(timezone.utc)
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_daily_reset(user_id, now=moment, lock=False)
            if profile.locked_until and profile.locked_until > moment:
                return
            limit = profile.daily_try_limit or self._daily_limit
            if profile.cycle_started_at is None:
                profile.cycle_started_at = moment
            profile.tries_used = max(profile.tries_used, 0) + 1
            if profile.tries_used == 1:
                profile.cycle_started_at = moment
            profile.daily_used = profile.tries_used
            if limit > 0 and profile.tries_used >= limit:
                profile.locked_until = moment + timedelta(hours=24)
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

    async def register_contact_generation(
        self,
        user_id: int,
        *,
        now: Optional[datetime] = None,
        initial_trigger: int,
        reminder_trigger: int,
    ) -> tuple[int, Optional[str]]:
        """Update per-day contact counters and return the active trigger."""

        moment = now or datetime.now(timezone.utc)
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            started = profile.contact_generations_started_at
            if not started or moment - started >= timedelta(hours=24):
                profile.contact_generations_started_at = moment
                profile.contact_generations_today = 0
                profile.contact_prompt_second_sent = False
                profile.contact_prompt_sixth_sent = False
                profile.contact_skip_once = False
            profile.contact_generations_today = max(
                profile.contact_generations_today, 0
            ) + 1
            trigger: Optional[str] = None
            if (
                profile.contact_generations_today == initial_trigger
                and not profile.contact_prompt_second_sent
            ):
                trigger = "second"
            elif (
                profile.contact_generations_today == reminder_trigger
                and not profile.contact_prompt_sixth_sent
            ):
                trigger = "sixth"
            await asyncio.to_thread(self._upsert_user, profile)
            return profile.contact_generations_today, trigger

    async def mark_contact_prompt_sent(
        self, user_id: int, trigger: str
    ) -> None:
        """Persist the fact that the contact prompt was shown for a trigger."""

        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            if trigger == "second":
                profile.contact_prompt_second_sent = True
            elif trigger == "sixth":
                profile.contact_prompt_sixth_sent = True
            else:
                return
            await asyncio.to_thread(self._upsert_user, profile)

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

    async def save_contact(
        self,
        user_id: int,
        phone_number: str,
        *,
        saved_at: Optional[datetime] = None,
    ) -> None:
        moment = saved_at or datetime.now(timezone.utc)
        await asyncio.to_thread(self._save_contact_sync, user_id, phone_number, moment)

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

    async def clear_seen_models(
        self, user_id: int, *, context: str | None = None
    ) -> None:
        await asyncio.to_thread(self._clear_seen_models_sync, user_id, context)

    async def reset_user_session(self, user_id: int) -> None:
        lock = self._ensure_lock()
        async with lock:
            profile = await self.ensure_user(user_id)
            if profile is None:
                return
            preserved_contact_never = profile.contact_never
            new_profile = UserProfile(
                user_id=user_id,
                gender=None,
                age_bucket=None,
                style="normal",
                daily_used=profile.daily_used,
                tries_used=profile.tries_used,
                daily_try_limit=profile.daily_try_limit,
                cycle_started_at=profile.cycle_started_at,
                cycle_index=profile.cycle_index,
                locked_until=profile.locked_until,
                nudge_sent_cycle=profile.nudge_sent_cycle,
                last_reset_at=profile.last_reset_at,
                seen_models=[],
                remind_at=None,
                referrer_id=profile.referrer_id,
                gen_count=0,
                contact_skip_once=False,
                contact_never=preserved_contact_never,
                last_activity_ts=profile.last_activity_ts,
                idle_reminder_sent=profile.idle_reminder_sent,
                social_ad_shown=profile.social_ad_shown,
                last_more_message_id=None,
                last_more_message_type=None,
                last_more_message_payload=None,
                contact_generations_today=0,
                contact_generations_started_at=None,
                contact_prompt_second_sent=False,
                contact_prompt_sixth_sent=False,
            )
            await asyncio.to_thread(self._upsert_user, new_profile)
            await asyncio.to_thread(self._clear_seen_models_sync, user_id, None)

    async def list_seen_models(self, user_id: int, *, context: str) -> set[str]:
        return await asyncio.to_thread(self._list_seen_models_sync, user_id, context)

    async def has_style_feedback(self, user_id: int) -> bool:
        return await asyncio.to_thread(self._has_style_feedback_sync, user_id)

    async def ensure_style_preferences(
        self, user_id: int, styles: Iterable[str], *, when: datetime | None = None
    ) -> None:
        unique = [style for style in dict.fromkeys(styles or []) if style]
        if not unique:
            return
        timestamp = (when or datetime.now(timezone.utc)).isoformat()
        await asyncio.to_thread(
            self._ensure_style_preferences_sync, user_id, unique, timestamp
        )

    async def list_style_preferences(
        self, user_id: int, *, styles: Iterable[str] | None = None
    ) -> dict[str, tuple[int, int]]:
        style_list = (
            [style for style in dict.fromkeys(styles) if style]
            if styles is not None
            else None
        )
        if style_list is not None and not style_list:
            return {}
        return await asyncio.to_thread(
            self._list_style_preferences_sync, user_id, style_list
        )

    async def insert_style_vote(
        self,
        user_id: int,
        generation_id: str,
        style: str,
        vote: str,
        *,
        created_at: datetime | None = None,
    ) -> bool:
        timestamp = (created_at or datetime.now(timezone.utc)).isoformat()
        return await asyncio.to_thread(
            self._insert_style_vote_sync,
            user_id,
            generation_id,
            style,
            vote,
            timestamp,
        )

    async def increment_style_preference(
        self,
        user_id: int,
        style: str,
        *,
        alpha_inc: int = 0,
        beta_inc: int = 0,
        updated_at: datetime | None = None,
    ) -> None:
        timestamp = (updated_at or datetime.now(timezone.utc)).isoformat()
        await asyncio.to_thread(
            self._increment_style_preference_sync,
            user_id,
            style,
            alpha_inc,
            beta_inc,
            timestamp,
        )

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

    async def set_last_more_message(
        self,
        user_id: int,
        message_id: int | None,
        message_type: str | None,
        payload: dict[str, Any] | None,
    ) -> None:
        profile = await self.ensure_user(user_id)
        profile.last_more_message_id = message_id
        profile.last_more_message_type = message_type
        profile.last_more_message_payload = payload
        await asyncio.to_thread(self._upsert_user, profile)

    async def list_due_reminders(self, now: datetime) -> List[UserProfile]:
        return await asyncio.to_thread(self._list_due_reminders_sync, now)

    # internal helpers
    def _create_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    gender TEXT,
                    age_bucket TEXT,
                    style TEXT,
                    daily_used INTEGER NOT NULL DEFAULT 0,
                    tries_used INTEGER NOT NULL DEFAULT 0,
                    daily_try_limit INTEGER NOT NULL DEFAULT {int(self._daily_limit)},
                    cycle_started_at TEXT,
                    cycle_index INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    nudge_sent_cycle INTEGER NOT NULL DEFAULT 0,
                    last_reset_at TEXT,
                    seen_models TEXT,
                    remind_at TEXT,
                    referrer_id INTEGER,
                    gen_count INTEGER NOT NULL DEFAULT 0,
                    contact_skip_once INTEGER NOT NULL DEFAULT 0,
                    contact_never INTEGER NOT NULL DEFAULT 0,
                    last_activity_ts INTEGER NOT NULL DEFAULT 0,
                    idle_reminder_sent INTEGER NOT NULL DEFAULT 0,
                    social_ad_shown INTEGER NOT NULL DEFAULT 0,
                    last_more_message_id INTEGER,
                    last_more_message_type TEXT,
                    last_more_message_payload TEXT,
                    contact_generations_today INTEGER NOT NULL DEFAULT 0,
                    contact_generations_started_at TEXT,
                    contact_prompt_second_sent INTEGER NOT NULL DEFAULT 0,
                    contact_prompt_sixth_sent INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_user_columns(conn)
            self._ensure_seen_table(conn)
            self._ensure_style_tables(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version_hash TEXT NOT NULL
                )
                """
            )
            self._ensure_contact_table(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contact_shares (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    phone_number TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _row_to_profile(self, row: sqlite3.Row) -> UserProfile:
        last_reset_at = (
            datetime.fromisoformat(row["last_reset_at"]) if row["last_reset_at"] else None
        )
        remind_at = datetime.fromisoformat(row["remind_at"]) if row["remind_at"] else None
        seen_models = json.loads(row["seen_models"]) if row["seen_models"] else []
        social_ad_raw = 0
        columns: Iterable[str] = ()
        if isinstance(row, sqlite3.Row):
            columns = row.keys()
            if "social_ad_shown" in columns:
                social_ad_raw = row["social_ad_shown"]
        else:
            try:
                social_ad_raw = row["social_ad_shown"]
            except Exception:  # pragma: no cover - defensive fallback
                social_ad_raw = 0

        def _has(column: str) -> bool:
            return column in columns if columns else False

        tries_used = row["tries_used"] if _has("tries_used") else row["daily_used"] or 0
        daily_limit = row["daily_try_limit"] if _has("daily_try_limit") else self._daily_limit
        cycle_started_at = (
            datetime.fromisoformat(row["cycle_started_at"])
            if _has("cycle_started_at") and row["cycle_started_at"]
            else None
        )
        locked_until = (
            datetime.fromisoformat(row["locked_until"])
            if _has("locked_until") and row["locked_until"]
            else None
        )
        cycle_index = row["cycle_index"] if _has("cycle_index") else 0
        nudge_sent = bool(row["nudge_sent_cycle"] or 0) if _has("nudge_sent_cycle") else False
        daily_used = row["daily_used"] if row["daily_used"] is not None else tries_used
        if daily_used != tries_used and tries_used >= 0:
            daily_used = tries_used
        payload_raw = row["last_more_message_payload"] if _has("last_more_message_payload") else None
        payload_dict = None
        if payload_raw:
            try:
                payload_dict = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload_dict = None
        contact_gen_started_at = None
        if _has("contact_generations_started_at") and row[
            "contact_generations_started_at"
        ]:
            contact_gen_started_at = datetime.fromisoformat(
                row["contact_generations_started_at"]
            )
        return UserProfile(
            user_id=row["user_id"],
            gender=row["gender"],
            age_bucket=row["age_bucket"],
            style=row["style"] or "normal",
            daily_used=daily_used,
            tries_used=max(int(tries_used), 0),
            daily_try_limit=int(daily_limit) if daily_limit is not None else self._daily_limit,
            cycle_started_at=cycle_started_at,
            cycle_index=int(cycle_index) if cycle_index is not None else 0,
            locked_until=locked_until,
            nudge_sent_cycle=nudge_sent,
            last_reset_at=last_reset_at,
            seen_models=seen_models,
            remind_at=remind_at,
            referrer_id=row["referrer_id"],
            gen_count=row["gen_count"] if row["gen_count"] is not None else 0,
            contact_skip_once=bool(row["contact_skip_once"] or 0),
            contact_never=bool(row["contact_never"] or 0),
            last_activity_ts=row["last_activity_ts"] if row["last_activity_ts"] else 0,
            idle_reminder_sent=bool(row["idle_reminder_sent"] or 0),
            social_ad_shown=bool(social_ad_raw or 0),
            last_more_message_id=(
                row["last_more_message_id"] if _has("last_more_message_id") else None
            ),
            last_more_message_type=(
                row["last_more_message_type"] if _has("last_more_message_type") else None
            ),
            last_more_message_payload=payload_dict,
            contact_generations_today=(
                row["contact_generations_today"]
                if _has("contact_generations_today") and row["contact_generations_today"]
                else 0
            ),
            contact_generations_started_at=contact_gen_started_at,
            contact_prompt_second_sent=bool(
                row["contact_prompt_second_sent"] or 0
            )
            if _has("contact_prompt_second_sent")
            else False,
            contact_prompt_sixth_sent=bool(
                row["contact_prompt_sixth_sent"] or 0
            )
            if _has("contact_prompt_sixth_sent")
            else False,
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
        tries_used = max(int(data.get("tries_used", 0) or 0), 0)
        data["tries_used"] = tries_used
        data["daily_used"] = tries_used
        limit_value = data.get("daily_try_limit")
        data["daily_try_limit"] = (
            int(limit_value)
            if isinstance(limit_value, int) and limit_value > 0
            else int(self._daily_limit)
        )
        payload_value = data.get("last_more_message_payload")
        if payload_value is None:
            data["last_more_message_payload"] = None
        elif isinstance(payload_value, str):
            data["last_more_message_payload"] = payload_value
        else:
            try:
                data["last_more_message_payload"] = json.dumps(payload_value)
            except TypeError:
                data["last_more_message_payload"] = None
        started_at = data.get("contact_generations_started_at")
        if started_at and not isinstance(started_at, str):
            data["contact_generations_started_at"] = started_at.isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    gender,
                    age_bucket,
                    style,
                    daily_used,
                    tries_used,
                    daily_try_limit,
                    cycle_started_at,
                    cycle_index,
                    locked_until,
                    nudge_sent_cycle,
                    last_reset_at,
                    seen_models,
                    remind_at,
                    referrer_id,
                    gen_count,
                    contact_skip_once,
                    contact_never,
                    last_activity_ts,
                    idle_reminder_sent,
                    social_ad_shown,
                    last_more_message_id,
                    last_more_message_type,
                    last_more_message_payload,
                    contact_generations_today,
                    contact_generations_started_at,
                    contact_prompt_second_sent,
                    contact_prompt_sixth_sent
                )
                VALUES (
                    :user_id,
                    :gender,
                    :age_bucket,
                    :style,
                    :daily_used,
                    :tries_used,
                    :daily_try_limit,
                    :cycle_started_at,
                    :cycle_index,
                    :locked_until,
                    :nudge_sent_cycle,
                    :last_reset_at,
                    :seen_models,
                    :remind_at,
                    :referrer_id,
                    :gen_count,
                    :contact_skip_once,
                    :contact_never,
                    :last_activity_ts,
                    :idle_reminder_sent,
                    :social_ad_shown,
                    :last_more_message_id,
                    :last_more_message_type,
                    :last_more_message_payload,
                    :contact_generations_today,
                    :contact_generations_started_at,
                    :contact_prompt_second_sent,
                    :contact_prompt_sixth_sent
                )
                ON CONFLICT(user_id) DO UPDATE SET
                    gender=excluded.gender,
                    age_bucket=excluded.age_bucket,
                    style=excluded.style,
                    daily_used=excluded.daily_used,
                    tries_used=excluded.tries_used,
                    daily_try_limit=excluded.daily_try_limit,
                    cycle_started_at=excluded.cycle_started_at,
                    cycle_index=excluded.cycle_index,
                    locked_until=excluded.locked_until,
                    nudge_sent_cycle=excluded.nudge_sent_cycle,
                    last_reset_at=excluded.last_reset_at,
                    seen_models=excluded.seen_models,
                    remind_at=excluded.remind_at,
                    referrer_id=excluded.referrer_id,
                    gen_count=excluded.gen_count,
                    contact_skip_once=excluded.contact_skip_once,
                    contact_never=excluded.contact_never,
                    last_activity_ts=excluded.last_activity_ts,
                    idle_reminder_sent=excluded.idle_reminder_sent,
                    social_ad_shown=excluded.social_ad_shown,
                    last_more_message_id=excluded.last_more_message_id,
                    last_more_message_type=excluded.last_more_message_type,
                    last_more_message_payload=excluded.last_more_message_payload,
                    contact_generations_today=excluded.contact_generations_today,
                    contact_generations_started_at=excluded.contact_generations_started_at,
                    contact_prompt_second_sent=excluded.contact_prompt_second_sent,
                    contact_prompt_sixth_sent=excluded.contact_prompt_sixth_sent
                """,
                {
                    "user_id": data["user_id"],
                    "gender": data["gender"],
                    "age_bucket": data["age_bucket"],
                    "style": data["style"],
                    "daily_used": data["daily_used"],
                    "tries_used": data["tries_used"],
                    "daily_try_limit": data["daily_try_limit"],
                    "cycle_started_at": (
                        data["cycle_started_at"].isoformat()
                        if data["cycle_started_at"]
                        else None
                    ),
                    "cycle_index": data["cycle_index"],
                    "locked_until": (
                        data["locked_until"].isoformat()
                        if data["locked_until"]
                        else None
                    ),
                    "nudge_sent_cycle": 1 if data["nudge_sent_cycle"] else 0,
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
                    "social_ad_shown": 1 if data["social_ad_shown"] else 0,
                    "last_more_message_id": data["last_more_message_id"],
                    "last_more_message_type": data["last_more_message_type"],
                    "last_more_message_payload": data["last_more_message_payload"],
                    "contact_generations_today": data["contact_generations_today"],
                    "contact_generations_started_at": data[
                        "contact_generations_started_at"
                    ],
                    "contact_prompt_second_sent": 1
                    if data["contact_prompt_second_sent"]
                    else 0,
                    "contact_prompt_sixth_sent": 1
                    if data["contact_prompt_sixth_sent"]
                    else 0,
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

    def _set_nudge_flag_sync(self, user_id: int, value: bool) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE users SET nudge_sent_cycle = ? WHERE user_id = ?",
                (1 if value else 0, user_id),
            )
            conn.commit()

    def _list_social_ad_candidates_sync(self, threshold_ts: int) -> List[UserProfile]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT * FROM users
                WHERE last_activity_ts > 0
                  AND last_activity_ts <= ?
                  AND social_ad_shown = 0
                """,
                (threshold_ts,),
            )
            rows = cur.fetchall()
        return [self._row_to_profile(row) for row in rows]

    def _mark_social_ad_shown_sync(self, user_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE users SET social_ad_shown = 1 WHERE user_id = ?",
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

    def _clear_seen_models_sync(
        self, user_id: int, context: str | None
    ) -> None:
        with self._connection() as conn:
            if context is None:
                conn.execute(
                    "DELETE FROM user_seen_models WHERE user_id = ?",
                    (user_id,),
                )
            else:
                conn.execute(
                    "DELETE FROM user_seen_models WHERE user_id = ? AND context = ?",
                    (user_id, context),
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

    def _has_style_feedback_sync(self, user_id: int) -> bool:
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM user_style_votes WHERE user_id = ? LIMIT 1",
                (user_id,),
            )
            return cur.fetchone() is not None

    def _ensure_style_preferences_sync(
        self, user_id: int, styles: list[str], timestamp: str
    ) -> None:
        if not styles:
            return
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO user_style_pref (
                    user_id,
                    style,
                    alpha,
                    beta,
                    updated_at
                )
                VALUES (?, ?, 1, 1, ?)
                """,
                [(user_id, style, timestamp) for style in styles],
            )
            conn.commit()

    def _list_style_preferences_sync(
        self, user_id: int, styles: list[str] | None
    ) -> dict[str, tuple[int, int]]:
        with self._connection() as conn:
            if styles is None:
                cur = conn.execute(
                    "SELECT style, alpha, beta FROM user_style_pref WHERE user_id = ?",
                    (user_id,),
                )
            else:
                placeholders = ",".join("?" for _ in styles)
                query = (
                    "SELECT style, alpha, beta FROM user_style_pref "
                    f"WHERE user_id = ? AND style IN ({placeholders})"
                )
                cur = conn.execute(query, (user_id, *styles))
            rows = cur.fetchall()
        return {row[0]: (int(row[1]), int(row[2])) for row in rows}

    def _insert_style_vote_sync(
        self,
        user_id: int,
        generation_id: str,
        style: str,
        vote: str,
        timestamp: str,
    ) -> bool:
        with self._connection() as conn:
            before = conn.total_changes
            conn.execute(
                """
                INSERT INTO user_style_votes (
                    user_id,
                    generation_id,
                    style,
                    vote,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, generation_id) DO NOTHING
                """,
                (user_id, generation_id, style, vote, timestamp),
            )
            inserted = conn.total_changes > before
            conn.commit()
            return inserted

    def _increment_style_preference_sync(
        self,
        user_id: int,
        style: str,
        alpha_inc: int,
        beta_inc: int,
        timestamp: str,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO user_style_pref (
                    user_id,
                    style,
                    alpha,
                    beta,
                    updated_at
                )
                VALUES (?, ?, 1, 1, ?)
                """,
                (user_id, style, timestamp),
            )
            conn.execute(
                """
                UPDATE user_style_pref
                SET alpha = alpha + ?,
                    beta = beta + ?,
                    updated_at = ?
                WHERE user_id = ? AND style = ?
                """,
                (alpha_inc, beta_inc, timestamp, user_id, style),
            )
            conn.commit()

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

    def _save_contact_sync(
        self, user_id: int, phone_number: str, when: datetime
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO contact_shares (user_id, phone_number, saved_at)
                VALUES (?, ?, ?)
                """,
                (user_id, phone_number, when.isoformat()),
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

    def _ensure_style_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_style_pref (
                user_id INTEGER NOT NULL,
                style TEXT NOT NULL,
                alpha INTEGER NOT NULL DEFAULT 1,
                beta INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, style)
            )
            """,
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_style_votes (
                user_id INTEGER NOT NULL,
                generation_id TEXT NOT NULL,
                style TEXT NOT NULL,
                vote TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, generation_id)
            )
            """,
        )

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
        if "social_ad_shown" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN social_ad_shown INTEGER NOT NULL DEFAULT 0"
            )
        if "last_more_message_id" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN last_more_message_id INTEGER")
        if "last_more_message_type" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN last_more_message_type TEXT")
        if "last_more_message_payload" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN last_more_message_payload TEXT")
        if "tries_used" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN tries_used INTEGER NOT NULL DEFAULT 0"
            )
        if "daily_try_limit" not in columns:
            conn.execute(
                f"ALTER TABLE users ADD COLUMN daily_try_limit INTEGER NOT NULL DEFAULT {int(self._daily_limit)}"
            )
        if "cycle_started_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN cycle_started_at TEXT")
        if "cycle_index" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN cycle_index INTEGER NOT NULL DEFAULT 0"
            )
        if "locked_until" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")
        if "nudge_sent_cycle" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN nudge_sent_cycle INTEGER NOT NULL DEFAULT 0"
            )
        if "contact_generations_today" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN contact_generations_today INTEGER NOT NULL DEFAULT 0"
            )
        if "contact_generations_started_at" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN contact_generations_started_at TEXT"
            )
        if "contact_prompt_second_sent" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN contact_prompt_second_sent INTEGER NOT NULL DEFAULT 0"
            )
        if "contact_prompt_sixth_sent" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN contact_prompt_sixth_sent INTEGER NOT NULL DEFAULT 0"
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

    def _start_new_cycle(self, profile: UserProfile, now: datetime) -> None:
        profile.cycle_index = (profile.cycle_index or 0) + 1
        profile.tries_used = 0
        profile.daily_used = 0
        profile.cycle_started_at = None
        profile.locked_until = None
        profile.nudge_sent_cycle = False
        profile.last_reset_at = now

    def _apply_profile_defaults(self, profile: UserProfile) -> bool:
        changed = False
        if profile.daily_try_limit <= 0:
            profile.daily_try_limit = self._daily_limit
            changed = True
        if profile.tries_used < 0:
            profile.tries_used = 0
            changed = True
        if profile.daily_used != profile.tries_used:
            profile.daily_used = profile.tries_used
            changed = True
        if profile.nudge_sent_cycle not in {True, False}:
            profile.nudge_sent_cycle = False
            changed = True
        return changed

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
