"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRT2CXRcmWxmWHKADYfHTadlxBUZ-"
    "R7nEX7HcAqrBo_PzSKYrCln4HFeCUJTB2q_C7asfwO7AOLNiwh/pub?output=csv"
)

DEFAULT_PRIVACY_POLICY_URL = "https://telegra.ph/Politika-konfidencialnosti-LOOV-10-29"


@dataclass(slots=True)
class CollageConfig:
    """Rendering options for dual-portrait collages."""

    slot_width: int
    slot_height: int
    separator_width: int
    padding: int
    separator_color: str
    background: str
    output_format: str
    jpeg_quality: int

    @property
    def width(self) -> int:
        """Total collage width including padding and separator."""

        return self.slot_width * 2 + self.separator_width + self.padding * 2

    @property
    def height(self) -> int:
        """Total collage height including padding."""

        return self.slot_height + self.padding * 2


@dataclass(slots=True)
class Config:
    """Top-level application configuration."""

    bot_token: str
    sheet_csv_url: str
    site_url: str
    privacy_policy_url: str
    promo_code: str
    daily_try_limit: int
    reminder_hours: int
    idle_reminder_minutes: int
    csv_fetch_ttl_sec: int
    csv_fetch_retries: int
    catalog_row_limit: int | None
    catalog_sheet_id: Optional[str]
    catalog_sheet_gid: Optional[str]
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
    promo_video_path: Path
    promo_video_enabled: bool
    promo_video_width: int | None
    promo_video_height: int | None


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


def _as_positive_int_or_none(value: Optional[str]) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = int(normalized)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


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
    google_sheet_url = _get("GOOGLE_SHEET_URL")
    catalog_csv_url = (
        _get("CATALOG_CSV_URL")
        or _get("SHEET_CSV_URL")
        or google_sheet_url
    )
    catalog_sheet_id = (
        _get("CATALOG_SHEET_ID")
        or _get("GOOGLE_SHEET_ID")
        or _get("SHEET_ID")
    )
    catalog_sheet_gid = (
        _get("CATALOG_SHEET_GID")
        or _get("SHEET_GID")
        or _get("GOOGLE_SHEET_GID")
    )
    row_limit_raw = _get("CATALOG_ROW_LIMIT")
    row_limit: int | None = None
    if row_limit_raw:
        try:
            parsed_limit = int(row_limit_raw)
        except ValueError:
            parsed_limit = 0
        if parsed_limit > 0:
            row_limit = parsed_limit

    collage = CollageConfig(
        slot_width=max(_as_int(_get("COLLAGE_SLOT_WIDTH", "1080"), 1080), 1),
        slot_height=max(_as_int(_get("COLLAGE_SLOT_HEIGHT", "1440"), 1440), 1),
        separator_width=max(_as_int(_get("COLLAGE_SEPARATOR_WIDTH", "24"), 24), 0),
        padding=max(_as_int(_get("COLLAGE_PADDING", "0"), 0), 0),
        separator_color=_get("COLLAGE_SEPARATOR_COLOR", "#2A2A2A") or "#2A2A2A",
        background=_get("COLLAGE_BACKGROUND", "#000000") or "#000000",
        output_format=(_get("COLLAGE_FORMAT", "PNG") or "PNG").upper(),
        jpeg_quality=_as_int(_get("COLLAGE_JPEG_QUALITY", "90"), 90),
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

    privacy_policy_url = (
        _get("PRIVACY_POLICY_URL", DEFAULT_PRIVACY_POLICY_URL)
        or DEFAULT_PRIVACY_POLICY_URL
    )

    contacts_sheet_url = google_sheet_url
    google_credentials_raw = _get("GOOGLE_SERVICE_ACCOUNT_JSON")
    google_credentials_path = (
        Path(google_credentials_raw)
        if google_credentials_raw
        else None
    )

    promo_video_path = _as_path(
        _get("PROMO_VIDEO_PATH", "video/promo_start.mp4"), "video/promo_start.mp4"
    )
    promo_video_enabled = _as_bool(_get("PROMO_VIDEO_ENABLED", "1"), True)
    promo_video_width_raw = _get("PROMO_VIDEO_WIDTH")
    promo_video_height_raw = _get("PROMO_VIDEO_HEIGHT")
    promo_video_width = _as_positive_int_or_none(promo_video_width_raw)
    promo_video_height = _as_positive_int_or_none(promo_video_height_raw)
    if promo_video_width_raw and promo_video_width is None:
        logger.warning("Invalid PROMO_VIDEO_WIDTH value %r; ignoring", promo_video_width_raw)
    if promo_video_height_raw and promo_video_height is None:
        logger.warning("Invalid PROMO_VIDEO_HEIGHT value %r; ignoring", promo_video_height_raw)
    logger.info(
        "Promo video config: path=%s, enabled=%s, width=%s, height=%s",
        promo_video_path,
        promo_video_enabled,
        promo_video_width,
        promo_video_height,
    )

    api_key = _get("NANOBANANA_API_KEY", required=True) or ""
    if not api_key.strip():
        raise RuntimeError("NANOBANANA_API_KEY is required")

    return Config(
        bot_token=_get("BOT_TOKEN", required=True),
        sheet_csv_url=(
            catalog_csv_url
            or _get("SHEET_CSV_URL", DEFAULT_SHEET_URL)
            or DEFAULT_SHEET_URL
        ),
        catalog_sheet_id=catalog_sheet_id,
        catalog_sheet_gid=catalog_sheet_gid,
        site_url=(site_url or "https://loov.ru/") if site_url is not None else "https://loov.ru/",
        privacy_policy_url=privacy_policy_url,
        promo_code=promo_code,
        daily_try_limit=_as_int(_get("DAILY_TRY_LIMIT", "7"), 7),
        reminder_hours=_as_int(_get("REMINDER_HOURS", "24"), 24),
        idle_reminder_minutes=_as_int(idle_timeout_raw, 5),
        csv_fetch_ttl_sec=_as_int(_get("CSV_FETCH_TTL_SEC", "60"), 60),
        csv_fetch_retries=_as_int(_get("CSV_FETCH_RETRIES", "3"), 3),
        catalog_row_limit=row_limit,
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
        promo_video_path=promo_video_path,
        promo_video_enabled=promo_video_enabled,
        promo_video_width=promo_video_width,
        promo_video_height=promo_video_height,
    )


__all__ = ["Config", "CollageConfig", "load_config"]

