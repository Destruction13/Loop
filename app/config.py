"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from urllib.parse import parse_qs, urlparse

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
    social_links: tuple[SocialLink, ...]
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


def _as_int(value: Optional[str], fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _parse_social_links(raw: Optional[str]) -> list[SocialLink]:
    if raw is None:
        return []
    payload = raw.strip()
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Invalid SOCIAL_LINKS_JSON value; ignoring")
        return []
    if not isinstance(data, list):
        logger.warning("SOCIAL_LINKS_JSON must be a JSON array")
        return []
    result: list[SocialLink] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        if not title or not url:
            continue
        result.append(SocialLink(title=title, url=url))
    if not result:
        logger.warning("SOCIAL_LINKS_JSON does not contain valid entries")
    return result


def load_config(env_file: str | None = None) -> Config:
    """Load configuration from the provided .env file (or default location)."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    google_sheet_url = _get("GOOGLE_SHEET_URL")
    sheet_csv_url = _get("SHEET_CSV_URL")
    if not sheet_csv_url:
        if google_sheet_url:
            sheet_csv_url = google_sheet_url
        else:
            raise RuntimeError(
                "Environment variable SHEET_CSV_URL is required when GOOGLE_SHEET_URL is absent"
            )

    catalog_sheet_id: Optional[str] = None
    catalog_sheet_gid: Optional[str] = None
    if google_sheet_url:
        catalog_sheet_id, catalog_sheet_gid = _extract_sheet_id_and_gid(google_sheet_url)

    row_limit_raw = _get("CATALOG_ROW_LIMIT")
    row_limit: int | None = None
    if row_limit_raw:
        try:
            parsed_limit = int(row_limit_raw)
        except ValueError:
            parsed_limit = 0
        if parsed_limit > 0:
            row_limit = parsed_limit

    promo_code = _get("PROMO_CODE") or ""
    daily_try_limit = _as_int(_get("DAILY_TRY_LIMIT"), 7)

    social_links_raw = _get("SOCIAL_LINKS_JSON")
    if social_links_raw is None:
        social_links = []
    else:
        parsed_links = _parse_social_links(social_links_raw)
        social_links = parsed_links if parsed_links else []

    pick_scheme_raw = _get("PICK_SCHEME")
    pick_scheme = (pick_scheme_raw or "UNIVERSAL").strip() or "UNIVERSAL"

    google_credentials_raw = _get("GOOGLE_SERVICE_ACCOUNT_JSON")
    google_credentials_path = Path(google_credentials_raw) if google_credentials_raw else None

    landing_url = _get("LANDING_URL")
    site_url = landing_url.strip() if landing_url else "https://loov.ru/"

    promo_contact_code = promo_code or "CONTACT1000"

    return Config(
        bot_token=_get("BOT_TOKEN", required=True),
        sheet_csv_url=sheet_csv_url,
        catalog_sheet_id=catalog_sheet_id,
        catalog_sheet_gid=catalog_sheet_gid,
        site_url=site_url,
        privacy_policy_url=DEFAULT_PRIVACY_POLICY_URL,
        promo_code=promo_code,
        daily_try_limit=daily_try_limit,
        reminder_hours=24,
        idle_reminder_minutes=5,
        csv_fetch_ttl_sec=60,
        csv_fetch_retries=3,
        catalog_row_limit=row_limit,
        uploads_root=Path("./uploads"),
        results_root=Path("./results"),
        button_title_max=28,
        nanobanana_api_key=_get("NANOBANANA_API_KEY", required=True),
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
    )


__all__ = ["Config", "CollageConfig", "SocialLink", "load_config"]

