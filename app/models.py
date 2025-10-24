"""Domain models used by the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class FilterOptions:
    """User-selected filters for catalog lookup."""

    gender: str
    age_bucket: str
    style: str


@dataclass(slots=True)
class ModelMeta:
    """Metadata for a catalog item."""

    title: str
    product_url: str
    shape: Optional[str] = None
    brand: Optional[str] = None


@dataclass(slots=True)
class ModelItem:
    """Catalog item representation."""

    model_id: str
    thumb_path: Path
    overlay_path: Optional[Path]
    meta: ModelMeta


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

    def remaining(self, limit: int) -> int:
        """Return remaining tries for the day."""

        return max(limit - self.daily_used, 0)
