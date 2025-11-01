"""Utilities for exporting captured leads to Google Sheets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


import re
from urllib.parse import urlparse, parse_qs

from logger import get_logger, log_event

def _extract_sheet_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    s = url_or_id.strip().strip('"').strip("'")
    # если уже чистый ID
    if "/" not in s and "?" not in s:
        return s
    try:
        u = urlparse(s)
        m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", u.path)
        return m.group(1) if m else None
    except Exception:
        return None


@dataclass(slots=True)
class LeadPayload:
    """Payload describing a captured lead."""

    tg_user_id: int
    phone_e164: str
    source: str
    consent_ts: int
    username: Optional[str]
    full_name: Optional[str]


class LeadsExporter:
    """Append captured leads to a Google Sheets worksheet."""
    
    def __init__(
        self,
        *,
        enabled: bool,
        sheet_name: str,
        promo_code: str,
        spreadsheet_id: Optional[str] = None,
        spreadsheet_url: Optional[str] = None,
        credentials_path: Optional[str | Path] = None,
    ) -> None:
        self._enabled = enabled
        self._sheet_name = sheet_name
        self._promo_code = promo_code
        raw_id = spreadsheet_id or _extract_sheet_id(spreadsheet_url)
        self._spreadsheet_id = (raw_id or "").strip().strip('"').strip("'")
        if credentials_path is None:
            creds_str = ""
        else:
            creds_str = str(credentials_path)
        self._credentials_path = creds_str.strip().strip('"').strip("'")
        self._logger = get_logger("leads.export")

    async def export_lead_to_sheet(self, payload: LeadPayload) -> bool:
        """Append the lead information to the configured sheet."""

        if not self._enabled:
            self._logger.debug(
                "Экспорт лидов отключён; пропускаем запись",
                extra={"stage": "EXPORT_LEAD_DISABLED"},
            )
            return False
        if not payload.phone_e164:
            self._logger.debug(
                "Экспорт лида пропущен: отсутствует номер телефона",
                extra={
                    "stage": "EXPORT_LEAD_SKIPPED",
                    "payload": {"user_id": payload.tg_user_id},
                },
            )
            return False
        if not self._spreadsheet_id or not self._credentials_path:
            self._logger.warning(
                "Не задан spreadsheet ID или путь к ключу — экспорт лидов пропущен",
                extra={
                    "stage": "EXPORT_LEAD_CONFIG_MISSING",
                    "payload": {"user_id": payload.tg_user_id},
                },
            )
            return False
        try:
            await asyncio.to_thread(self._append_row, payload)
        except ImportError as exc:
            self._logger.error(
                "Библиотеки для экспорта лидов не установлены: %s",
                exc,
                extra={
                    "stage": "EXPORT_LEAD_ERROR",
                    "payload": {"user_id": payload.tg_user_id},
                },
            )
            return False
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Не удалось записать лид в таблицу: %s",
                exc,
                extra={
                    "stage": "EXPORT_LEAD_ERROR",
                    "payload": {"user_id": payload.tg_user_id},
                },
            )
            return False
        log_event(
            "INFO",
            "leads.export",
            "📤 Экспорт лида завершён",
            user_id=payload.tg_user_id,
            stage="EXPORT_LEAD",
            extra={"sheet": self._sheet_name},
        )
        return True

    def _append_row(self, payload: LeadPayload) -> None:
        import gspread  # type: ignore[import-not-found]
        from google.oauth2.service_account import (  # type: ignore[import-not-found]
            Credentials,
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ]
        credentials = Credentials.from_service_account_file(
            self._credentials_path, scopes=scopes
        )
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(self._spreadsheet_id)
        try:
            worksheet = spreadsheet.worksheet(self._sheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=self._sheet_name, rows=10, cols=10)
        timestamp = datetime.fromtimestamp(payload.consent_ts, tz=timezone.utc).isoformat()
        row = [
            timestamp,
            str(payload.tg_user_id),
            payload.username or "",
            payload.full_name or "",
            payload.phone_e164,
            payload.source,
            self._promo_code,
        ]
        worksheet.append_row(row, value_input_option="USER_ENTERED")


__all__ = ["LeadsExporter", "LeadPayload"]
