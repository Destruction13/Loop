"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from app.analytics.constants import (
    ANALYTICS_SHEET_DEFAULT,
    DASHBOARD_SHEET_DEFAULT,
    EVENTS_SHEET_DEFAULT,
)

logger = logging.getLogger(__name__)


def _extract_sheet_id_and_gid(url_or_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Принимает либо чистый ID (1VlHdgf...FtY), либо любой Google Sheets URL.
    Возвращает (sheet_id, gid) или (None, None), если ничего не нашли.
    Поддерживает обычные edit-URL с /spreadsheets/d/<id>/ и вытаскивает ?gid=...
    Для опубликованных CSV ссылок /e/2PACX... извлечь sheet_id нельзя — вернёт None.
    """
    if not url_or_id:
        return None, None
    s = url_or_id.strip().strip('"').strip("'")
    # Если это уже похоже на голый ID (нет слешей и вопросиков) — вернём как есть
    if "/" not in s and "?" not in s:
        return s, None

    try:
        u = urlparse(s)
        # /spreadsheets/d/<id>...
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", u.path)
        sheet_id = m.group(1) if m else None
        # ?gid=123456
        q = parse_qs(u.query)
        gid = (q.get("gid") or [None])[0]
        return sheet_id, gid
    except Exception:
        return None, None
DEFAULT_PRIVACY_POLICY_URL = "https://telegra.ph/Politika-konfidencialnosti-LOOV-10-29"
DEFAULT_LANDING_URL = "https://loov.ru/"


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


@dataclass(frozen=True, slots=True)
class SocialLink:
    """Represents a social media link displayed to the user."""

    title: str
    url: str


@dataclass(frozen=True, slots=True)
class NanoBananaKeySlot:
    """Configured NanoBanana API key slot."""

    name: str
    api_key: str


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
    nanobanana_key_slots: tuple[NanoBananaKeySlot, ...]
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
    social_links: tuple[SocialLink, ...]
    contacts_sheet_url: Optional[str]
    google_service_account_json: Optional[Path]
    promo_video_path: Path
    promo_video_enabled: bool
    promo_video_width: int | None
    promo_video_height: int | None
    analytics_events_sheet_name: str
    analytics_sheet_name: str
    analytics_dashboard_sheet_name: str
    analytics_flush_interval_sec: int
    analytics_spreadsheet_id: str | None


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is required")
    stripped = value.strip()
    if not stripped:
        raise RuntimeError(f"Environment variable {name} must not be empty")
    return stripped


def _optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped and default is None:
        return None
    if not stripped:
        return default
    return stripped


def _require_url(name: str) -> str:
    value = _require_env(name)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"{name} must be a valid HTTP(S) URL")
    return value


def _parse_int_env(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = _optional_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer value") from None
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be greater than or equal to {minimum}")
    return value


def _parse_social_links(raw: Optional[str]) -> list[SocialLink]:
    if raw is None:
        return []
    payload = raw.strip()
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise RuntimeError("SOCIAL_LINKS_JSON must be valid JSON") from None
    if not isinstance(data, list):
        raise RuntimeError("SOCIAL_LINKS_JSON must be a JSON array")
    result: list[SocialLink] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise RuntimeError("SOCIAL_LINKS_JSON entries must be objects with title and url")
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        if not title or not url:
            raise RuntimeError("Each SOCIAL_LINKS_JSON entry must contain title and url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("SOCIAL_LINKS_JSON url values must be valid HTTP(S) links")
        result.append(SocialLink(title=title, url=url))
    return result


_KEY_SLOT_PATTERN = re.compile(r"^K_(\d+)$")


def _load_nanobanana_key_slots() -> tuple[NanoBananaKeySlot, ...]:
    """Return configured NanoBanana key slots sorted by their numeric suffix."""

    slots: list[tuple[int, NanoBananaKeySlot]] = []
    for name, value in os.environ.items():
        match = _KEY_SLOT_PATTERN.fullmatch(name)
        if not match:
            continue
        sanitized = (value or "").strip()
        if not sanitized:
            continue
        index = int(match.group(1))
        slots.append((index, NanoBananaKeySlot(name=name, api_key=sanitized)))

    slots.sort(key=lambda item: item[0])
    ordered_slots = tuple(slot for _, slot in slots)
    if not ordered_slots:
        raise RuntimeError(
            "At least one NanoBanana API key (K_1…K_N) must be provided via environment variables"
        )
    return ordered_slots


def load_config(env_file: str | None = None) -> Config:
    """Load configuration from the provided .env file (or default location)."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    bot_token = _require_env("BOT_TOKEN")
    sheet_csv_url_override = _optional_env("SHEET_CSV_URL")
    google_sheet_url = _require_url("GOOGLE_SHEET_URL")
    sheet_csv_url = (
        _require_url("SHEET_CSV_URL") if sheet_csv_url_override else google_sheet_url
    )
    landing_url_raw = _optional_env("LANDING_URL", DEFAULT_LANDING_URL) or DEFAULT_LANDING_URL
    landing_parsed = urlparse(landing_url_raw)
    if landing_parsed.scheme not in {"http", "https"} or not landing_parsed.netloc:
        raise RuntimeError("LANDING_URL must be a valid HTTP(S) URL")
    landing_url = landing_url_raw
    social_links_json = _optional_env("SOCIAL_LINKS_JSON", "[]") or "[]"
    social_links = _parse_social_links(social_links_json)
    nanobanana_key_slots = _load_nanobanana_key_slots()
    promo_code = _optional_env("PROMO_CODE", "") or ""
    daily_try_limit = _parse_int_env("DAILY_TRY_LIMIT", 7, minimum=1)
    catalog_row_raw = _parse_int_env("CATALOG_ROW_LIMIT", 0, minimum=0)
    catalog_row_limit = catalog_row_raw or None
    pick_scheme_raw = _optional_env("PICK_SCHEME", "UNIVERSAL") or "UNIVERSAL"
    pick_scheme = pick_scheme_raw.strip() or "UNIVERSAL"
    google_credentials_raw = _optional_env("GOOGLE_SERVICE_ACCOUNT_JSON")
    google_credentials_path = Path(google_credentials_raw) if google_credentials_raw else None

    catalog_sheet_id, catalog_sheet_gid = _extract_sheet_id_and_gid(google_sheet_url)

    promo_contact_code = promo_code or "CONTACT1000"

    analytics_events_sheet_name = (
        _optional_env("ANALYTICS_EVENTS_SHEET_NAME", EVENTS_SHEET_DEFAULT)
        or EVENTS_SHEET_DEFAULT
    )
    analytics_sheet_name = (
        _optional_env("ANALYTICS_SHEET_NAME", ANALYTICS_SHEET_DEFAULT)
        or ANALYTICS_SHEET_DEFAULT
    )
    analytics_dashboard_sheet_name = (
        _optional_env("ANALYTICS_DASHBOARD_SHEET_NAME", DASHBOARD_SHEET_DEFAULT)
        or DASHBOARD_SHEET_DEFAULT
    )
    analytics_flush_interval_sec = _parse_int_env(
        "ANALYTICS_FLUSH_INTERVAL_SEC", 30, minimum=5
    )
    analytics_spreadsheet_id = (_optional_env("LOG_SHEET_ID") or "").strip() or None

    return Config(
        bot_token=bot_token,
        sheet_csv_url=sheet_csv_url,
        catalog_sheet_id=catalog_sheet_id,
        catalog_sheet_gid=catalog_sheet_gid,
        site_url=landing_url,
        privacy_policy_url=DEFAULT_PRIVACY_POLICY_URL,
        promo_code=promo_code,
        daily_try_limit=daily_try_limit,
        reminder_hours=24,
        idle_reminder_minutes=5,
        csv_fetch_ttl_sec=60,
        csv_fetch_retries=3,
        catalog_row_limit=catalog_row_limit,
        uploads_root=Path("./uploads"),
        results_root=Path("./results"),
        button_title_max=28,
        nanobanana_key_slots=nanobanana_key_slots,
        collage=CollageConfig(
            slot_width=1080,
            slot_height=1440,
            separator_width=24,
            padding=0,
            separator_color="#2A2A2A",
            background="#000000",
            output_format="PNG",
            jpeg_quality=90,
        ),
        batch_size=2,
        batch_layout_cols=2,
        pick_scheme=pick_scheme,
        reco_clear_on_catalog_change=True,
        reco_no_more_key="all_seen",
        contact_reward_rub=1000,
        promo_contact_code=promo_contact_code,
        leads_sheet_name="Leads",
        enable_leads_export=True,
        enable_idle_reminder=True,
        social_ad_minutes=20,
        enable_social_ad=True,
        social_links=tuple(social_links),
        contacts_sheet_url=google_sheet_url,
        google_service_account_json=google_credentials_path,
        promo_video_path=Path("video/promo_start.mp4"),
        promo_video_enabled=True,
        promo_video_width=None,
        promo_video_height=None,
        analytics_events_sheet_name=analytics_events_sheet_name,
        analytics_sheet_name=analytics_sheet_name,
        analytics_dashboard_sheet_name=analytics_dashboard_sheet_name,
        analytics_flush_interval_sec=analytics_flush_interval_sec,
        analytics_spreadsheet_id=analytics_spreadsheet_id,
    )


__all__ = ["Config", "CollageConfig", "SocialLink", "load_config"]

