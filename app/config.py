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
    """Rendering options for horizontal collages."""

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
    site_url: str
    promo_code: str
    daily_try_limit: int
    reminder_hours: int
    idle_reminder_minutes: int
    csv_fetch_ttl_sec: int
    csv_fetch_retries: int
    uploads_root: Path
    results_root: Path
    button_title_max: int
    nanobanana_api_key: str
    collage: CollageConfig
    batch_size: int
    batch_layout_cols: int
    pick_scheme: str
    reco_clear_on_catalog_change: bool
    reco_no_more_key: str
    contact_reward_rub: int
    promo_contact_code: str
    leads_sheet_name: str
    enable_leads_export: bool
    enable_idle_reminder: bool
    social_ad_minutes: int
    enable_social_ad: bool
    social_instagram_url: str
    social_tiktok_url: str
    contacts_sheet_url: Optional[str]
    google_service_account_json: Optional[Path]


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

    batch_size = max(_as_int(_get("BATCH_SIZE", "2"), 2), 1)
    batch_columns = max(_as_int(_get("BATCH_LAYOUT_COLS", "2"), 2), 1)
    collage = CollageConfig(
        width=_as_int(_get("CANVAS_WIDTH", "1600"), 1600),
        height=_as_int(_get("CANVAS_HEIGHT", "800"), 800),
        columns=batch_columns,
        margin=_as_int(_get("TILE_MARGIN", "30"), 30),
        divider_width=_as_int(_get("DIVIDER_WIDTH", "6"), 6),
        divider_color=_get("DIVIDER_COLOR", "#E5E5E5") or "#E5E5E5",
        background=_get("CANVAS_BG", "#FFFFFF") or "#FFFFFF",
        jpeg_quality=_as_int(_get("JPEG_QUALITY", "88"), 88),
    )

    promo_code = _get("PROMO_CODE", "DEMO 10") or "DEMO 10"
    promo_contact_raw = _get("PROMO_CONTACT_CODE")
    if promo_contact_raw is None:
        promo_contact_code = promo_code or "CONTACT1000"
    else:
        promo_contact_code = promo_contact_raw or (promo_code or "CONTACT1000")
    contact_reward_rub = _as_int(_get("CONTACT_REWARD_RUB", "1000"), 1000)
    leads_sheet_name = _get("LEADS_SHEET_NAME", "Leads") or "Leads"
    enable_leads_export = _as_bool(_get("ENABLE_LEADS_EXPORT", "1"), True)

    site_url = _get("SITE_URL")
    if site_url is None:
        site_url = _get("LANDING_URL")

    idle_timeout_raw = _get("AFK_SITE")
    if idle_timeout_raw is None:
        idle_timeout_raw = _get("IDLE_REMINDER_MINUTES")

    contacts_sheet_url = _get("GOOGLE_SHEET_URL")
    google_credentials_raw = _get("GOOGLE_SERVICE_ACCOUNT_JSON")
    google_credentials_path = (
        Path(google_credentials_raw)
        if google_credentials_raw
        else None
    )

    api_key = _get("NANOBANANA_API_KEY", required=True) or ""
    if not api_key.strip():
        raise RuntimeError("NANOBANANA_API_KEY is required")

    return Config(
        bot_token=_get("BOT_TOKEN", required=True),
        sheet_csv_url=_get("SHEET_CSV_URL", DEFAULT_SHEET_URL) or DEFAULT_SHEET_URL,
        site_url=(site_url or "https://loov.ru/") if site_url is not None else "https://loov.ru/",
        promo_code=promo_code,
        daily_try_limit=_as_int(_get("DAILY_TRY_LIMIT", "7"), 7),
        reminder_hours=_as_int(_get("REMINDER_HOURS", "24"), 24),
        idle_reminder_minutes=_as_int(idle_timeout_raw, 5),
        csv_fetch_ttl_sec=_as_int(_get("CSV_FETCH_TTL_SEC", "60"), 60),
        csv_fetch_retries=_as_int(_get("CSV_FETCH_RETRIES", "3"), 3),
        uploads_root=_as_path(_get("UPLOADS_ROOT", "./uploads"), "./uploads"),
        results_root=_as_path(_get("RESULTS_ROOT", "./results"), "./results"),
        button_title_max=_as_int(_get("BUTTON_TITLE_MAX", "28"), 28),
        nanobanana_api_key=api_key.strip(),
        collage=collage,
        batch_size=batch_size,
        batch_layout_cols=batch_columns,
        pick_scheme=_get("PICK_SCHEME", "GENDER_OR_GENDER_UNISEX")
        or "GENDER_OR_GENDER_UNISEX",
        reco_clear_on_catalog_change=_as_bool(_get("RECO_CLEAR_ON_CATALOG_CHANGE", "1"), True),
        reco_no_more_key=_get("MSG_NO_MORE_KEY", "all_seen") or "all_seen",
        contact_reward_rub=contact_reward_rub,
        promo_contact_code=promo_contact_code,
        leads_sheet_name=leads_sheet_name,
        enable_leads_export=enable_leads_export,
        enable_idle_reminder=_as_bool(_get("ENABLE_IDLE_REMINDER", "1"), True),
        social_ad_minutes=_as_int(_get("SOCIAL_AD_MINUTES", "20"), 20),
        enable_social_ad=_as_bool(_get("ENABLE_SOCIAL_AD", "1"), True),
        social_instagram_url=_get("SOCIAL_INSTAGRAM_URL", "https://instagram.com/loov")
        or "https://instagram.com/loov",
        social_tiktok_url=_get("SOCIAL_TIKTOK_URL", "https://tiktok.com/@loov")
        or "https://tiktok.com/@loov",
        contacts_sheet_url=contacts_sheet_url,
        google_service_account_json=google_credentials_path,
    )


__all__ = ["Config", "CollageConfig", "load_config"]
