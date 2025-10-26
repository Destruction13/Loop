"""Utilities for exporting captured leads to Google Sheets."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


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
        credentials_path: Optional[str] = None,
    ) -> None:
        self._enabled = enabled
        self._sheet_name = sheet_name
        self._promo_code = promo_code
        self._spreadsheet_id = spreadsheet_id or os.getenv("LEADS_SPREADSHEET_ID")
        creds_env = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self._credentials_path = credentials_path or creds_env
        self._logger = logging.getLogger("loop_bot.leads_export")

    async def export_lead_to_sheet(self, payload: LeadPayload) -> bool:
        """Append the lead information to the configured sheet."""

        if not self._enabled:
            self._logger.debug("Lead export disabled; skipping")
            return False
        if not payload.phone_e164:
            self._logger.debug("Lead export skipped due to empty phone")
            return False
        if not self._spreadsheet_id or not self._credentials_path:
            self._logger.warning(
                "Lead export skipped: missing spreadsheet id or credentials path"
            )
            return False
        try:
            await asyncio.to_thread(self._append_row, payload)
        except ImportError as exc:
            self._logger.error("Leads exporter missing dependency: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to export lead to sheet: %s", exc)
            return False
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
