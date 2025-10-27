"""Domain models used by the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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
    last_reset_at: Optional[datetime] = None
    seen_models: list[str] = field(default_factory=list)
    remind_at: Optional[datetime] = None
    referrer_id: Optional[int] = None
    gen_count: int = 0
    contact_skip_once: bool = False
    contact_never: bool = False
    last_activity_ts: int = 0
    idle_reminder_sent: bool = False

    def remaining(self, limit: int) -> int:
        """Return remaining tries for the day."""

        return max(limit - self.daily_used, 0)


@dataclass(slots=True)
class UserContact:
    """Stored phone number for a Telegram user."""

    tg_user_id: int
    phone_e164: str
    source: str
    consent: bool
    consent_ts: int
    reward_granted: bool = False
