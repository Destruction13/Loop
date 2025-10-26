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
    """Rendering options for horizontal three-tile collages."""

    width: int
    height: int
    columns: int
    margin: int
    divider_width: int
    divider_color: str
    background: str
    jpeg_quality: int


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
    batch_size: int
    batch_layout_cols: int
    pick_rule: str
    reco_unique_scope: str
    reco_clear_on_catalog_change: bool
    reco_topup_from_any: bool
    reco_no_more_key: str


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


def _as_unique_scope(value: Optional[str], fallback: str) -> str:
    allowed = {"24h", "7d", "all"}
    if not value:
        return fallback
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    return fallback


def load_config(env_file: str | None = None) -> Config:
    """Load configuration from the provided .env file (or default location)."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    batch_size = _as_int(_get("BATCH_SIZE", "3"), 3)
    batch_columns = _as_int(_get("BATCH_LAYOUT_COLS", "3"), 3)
    collage = CollageConfig(
        width=_as_int(_get("CANVAS_WIDTH", "1800"), 1800),
        height=_as_int(_get("CANVAS_HEIGHT", "600"), 600),
        columns=max(batch_columns, 1),
        margin=_as_int(_get("TILE_MARGIN", "30"), 30),
        divider_width=_as_int(_get("DIVIDER_WIDTH", "4"), 4),
        divider_color=_get("DIVIDER_COLOR", "#E5E5E5") or "#E5E5E5",
        background=_get("CANVAS_BG", "#FFFFFF") or "#FFFFFF",
        jpeg_quality=_as_int(_get("JPEG_QUALITY", "88"), 88),
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
        batch_size=max(batch_size, 1),
        batch_layout_cols=max(batch_columns, 1),
        pick_rule=_get("PICK_RULE", "2_1") or "2_1",
        reco_unique_scope=_as_unique_scope(_get("RECO_UNIQUE_SCOPE", "24h"), "24h"),
        reco_clear_on_catalog_change=_as_bool(_get("RECO_CLEAR_ON_CATALOG_CHANGE", "1"), True),
        reco_topup_from_any=_as_bool(_get("RECO_TOPUP_FROM_ANY", "0"), False),
        reco_no_more_key=_get("MSG_NO_MORE_KEY", "all_seen") or "all_seen",
    )


__all__ = ["Config", "CollageConfig", "load_config"]
