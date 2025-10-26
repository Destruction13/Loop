from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gspread

from db.init import Database

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SheetsConfig:
    spreadsheet_id: str
    service_account_file: Path
    interval_seconds: int

    @property
    def enabled(self) -> bool:
        return bool(self.spreadsheet_id and self.service_account_file.exists())


class SheetsExporter:
    def __init__(self, config: SheetsConfig) -> None:
        self._config = config
        self._client: gspread.Client | None = None
        self._last_event_id: int | None = None
        self._events_initialized = False

    async def run(self, db: Database, stop_event: asyncio.Event) -> None:
        if not self._config.enabled:
            LOGGER.info("Sheets export disabled")
            await stop_event.wait()
            return
        LOGGER.info("Sheets export started")
        try:
            while not stop_event.is_set():
                try:
                    await self._export(db)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Sheets export failed: %s", exc)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self._config.interval_seconds)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise

    async def _export(self, db: Database) -> None:
        users = await db.fetch_users()
        events = await db.fetch_events_since(self._last_event_id)
        await asyncio.to_thread(self._write_users, users)
        if events:
            await asyncio.to_thread(self._append_events, events)
            self._last_event_id = events[-1]["id"]

    def _client_instance(self) -> gspread.Client:
        if self._client is None:
            self._client = gspread.service_account(filename=str(self._config.service_account_file))
        return self._client

    def _worksheet(self, name: str) -> gspread.Worksheet:
        client = self._client_instance()
        spreadsheet = client.open_by_key(self._config.spreadsheet_id)
        try:
            return spreadsheet.worksheet(name)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=name, rows=100, cols=10)

    def _write_users(self, users: list[dict[str, Any]]) -> None:
        ws = self._worksheet("users")
        header = ["tg_id", "username", "name", "gender", "phone", "created_at", "updated_at"]
        rows = [
            [
                user.get("tg_id"),
                user.get("username"),
                user.get("name"),
                user.get("gender"),
                user.get("phone"),
                user.get("created_at"),
                user.get("updated_at"),
            ]
            for user in users
        ]
        ws.clear()
        ws.append_row(header)
        if rows:
            ws.append_rows(rows, value_input_option="USER_ENTERED")

    def _append_events(self, events: list[dict[str, Any]]) -> None:
        ws = self._worksheet("events")
        if not self._events_initialized:
            header = ["id", "tg_id", "type", "payload", "ts"]
            existing = ws.get_all_values()
            if not existing:
                ws.append_row(header)
            self._events_initialized = True
        rows = [
            [
                event.get("id"),
                event.get("tg_id"),
                event.get("type"),
                event.get("payload_json")
                if isinstance(event.get("payload_json"), str)
                else json.dumps(event.get("payload_json", {}), ensure_ascii=False),
                event.get("ts"),
            ]
            for event in events
        ]
        if rows:
            ws.append_rows(rows, value_input_option="USER_ENTERED")


__all__ = ["SheetsConfig", "SheetsExporter"]
