"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRT2CXRcmWxmWHKADYfHTadlxBUZ-"
    "R7nEX7HcAqrBo_PzSKYrCln4HFeCUJTB2q_C7asfwO7AOLNiwh/pub?output=csv"
)


@dataclass(slots=True)
class CollageConfig:
    """Rendering options for collage previews."""

    enabled: bool
    max_width: int
    padding_px: int
    cache_ttl_sec: int
    draw_divider: bool


@dataclass(slots=True)
class Config:
    """Top-level application configuration."""

    bot_token: str
    sheet_csv_url: str
    landing_url: str
    promo_code: str
    daily_try_limit: int
    reminder_hours: int
    csv_fetch_ttl_sec: int
    csv_fetch_retries: int
    mock_tryon: bool
    uploads_root: Path
    results_root: Path
    button_title_max: int
    nano_api_url: Optional[str]
    nano_api_key: Optional[str]
    collage: CollageConfig


def _get(name: str, default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if value is None and required:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _as_bool(value: Optional[str], fallback: bool) -> bool:
    if not value:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return fallback


def _as_int(value: Optional[str], fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_path(value: Optional[str], fallback: str) -> Path:
    return Path(value or fallback)


def load_config(env_file: str | None = None) -> Config:
    """Load configuration from the provided .env file (or default location)."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    collage = CollageConfig(
        enabled=_as_bool(_get("COLLAGE_ENABLED", "1"), True),
        max_width=_as_int(_get("COLLAGE_MAX_WIDTH", "1280"), 1280),
        padding_px=_as_int(_get("COLLAGE_PADDING_PX", "20"), 20),
        cache_ttl_sec=_as_int(_get("COLLAGE_CACHE_TTL_SEC", "300"), 300),
        draw_divider=_as_bool(_get("COLLAGE_DRAW_DIVIDER", "1"), True),
    )

    return Config(
        bot_token=_get("BOT_TOKEN", required=True),
        sheet_csv_url=_get("SHEET_CSV_URL", DEFAULT_SHEET_URL) or DEFAULT_SHEET_URL,
        landing_url=_get("LANDING_URL", "https://example.com/booking") or "https://example.com/booking",
        promo_code=_get("PROMO_CODE", "DEMO 10") or "DEMO 10",
        daily_try_limit=_as_int(_get("DAILY_TRY_LIMIT", "7"), 7),
        reminder_hours=_as_int(_get("REMINDER_HOURS", "24"), 24),
        csv_fetch_ttl_sec=_as_int(_get("CSV_FETCH_TTL_SEC", "60"), 60),
        csv_fetch_retries=_as_int(_get("CSV_FETCH_RETRIES", "3"), 3),
        mock_tryon=_as_bool(_get("MOCK_TRYON", "1"), True),
        uploads_root=_as_path(_get("UPLOADS_ROOT", "./uploads"), "./uploads"),
        results_root=_as_path(_get("RESULTS_ROOT", "./results"), "./results"),
        button_title_max=_as_int(_get("BUTTON_TITLE_MAX", "28"), 28),
        nano_api_url=_get("NANO_API_URL"),
        nano_api_key=_get("NANO_API_KEY"),
        collage=collage,
    )


__all__ = ["Config", "CollageConfig", "load_config"]
