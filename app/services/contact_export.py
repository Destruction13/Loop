"""Google Sheets exporter for Telegram contacts."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ContactRecord:
    """Structured contact information ready for export."""

    first_name: str
    phone_number: str
    telegram_link: str
    gender: str


class ContactSheetExporter:
    """Append Telegram contacts to a Google Sheets worksheet."""

    def __init__(
        self,
        *,
        sheet_url: str,
        worksheet_name: str,
        credentials_path: Optional[Path],
    ) -> None:
        self._sheet_url = sheet_url.strip()
        self._worksheet_name = worksheet_name
        self._credentials_path = credentials_path
        self._logger = logging.getLogger("loop_bot.contact_export")

    async def export_contact(self, record: ContactRecord) -> bool:
        """Append the provided contact to the configured worksheet."""

        if not self._sheet_url:
            self._logger.debug("Contact export skipped: sheet url not configured")
            return False
        if not self._credentials_path:
            self._logger.warning("Contact export skipped: credentials path missing")
            return False
        if not record.phone_number:
            self._logger.debug("Contact export skipped: empty phone number")
            return False
        try:
            await asyncio.to_thread(self._append_row, record)
        except ImportError as exc:
            self._logger.error("Contact export missing dependency: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to export contact to sheet: %s", exc)
            return False
        return True

    def _append_row(self, record: ContactRecord) -> None:
        import gspread  # type: ignore[import-not-found]
        from google.oauth2.service_account import (  # type: ignore[import-not-found]
            Credentials,
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ]
        credentials = Credentials.from_service_account_file(
            str(self._credentials_path), scopes=scopes
        )
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_url(self._sheet_url)
        try:
            worksheet = spreadsheet.worksheet(self._worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=self._worksheet_name, rows=10, cols=10
            )
        worksheet.append_row(
            [
                record.first_name,
                record.phone_number,
                record.telegram_link,
                record.gender,
            ],
            value_input_option="USER_ENTERED",
        )


__all__ = ["ContactSheetExporter", "ContactRecord"]
