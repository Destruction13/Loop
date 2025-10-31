"""Domain models used by the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(slots=True)
class FilterOptions:
    """User-selected filters for catalog lookup."""

    gender: str


@dataclass(slots=True)
class GlassModel:
    """Catalog entry parsed from Google Sheets."""

    unique_id: str
    title: str
    model_code: str
    site_url: str
    img_user_url: str
    img_nano_url: str
    gender: str


@dataclass(slots=True)
class UserProfile:
    """User preferences stored in repository."""

    user_id: int
    gender: Optional[str] = None
    age_bucket: Optional[str] = None
    style: str = "normal"
    daily_used: int = 0
    tries_used: int = 0
    daily_try_limit: int = 0
    cycle_started_at: Optional[datetime] = None
    cycle_index: int = 0
    locked_until: Optional[datetime] = None
    nudge_sent_cycle: bool = False
    last_reset_at: Optional[datetime] = None
    seen_models: list[str] = field(default_factory=list)
    remind_at: Optional[datetime] = None
    referrer_id: Optional[int] = None
    gen_count: int = 0
    contact_skip_once: bool = False
    contact_never: bool = False
    last_activity_ts: int = 0
    idle_reminder_sent: bool = False
    social_ad_shown: bool = False
    last_more_message_id: Optional[int] = None
    last_more_message_type: Optional[str] = None
    last_more_message_payload: Optional[dict[str, Any]] = None

    def remaining(self, limit: int | None = None) -> int:
        """Return remaining tries respecting the current cycle state."""

        effective_limit = limit if limit and limit > 0 else self.daily_try_limit
        if effective_limit <= 0:
            return 0
        if self.locked_until and self.locked_until > datetime.now(timezone.utc):
            return 0
        used = self.tries_used if self.tries_used >= 0 else 0
        return max(effective_limit - used, 0)


@dataclass(slots=True)
class UserContact:
    """Stored phone number for a Telegram user."""

    tg_user_id: int
    phone_e164: str
    source: str
    consent: bool
    consent_ts: int
    reward_granted: bool = False
