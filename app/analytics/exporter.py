"""Фоновый экспорт событий в Google Sheets и обновление KPI."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from logger import get_logger

from . import metrics, track
from .constants import ANALYTICS_HEADER, EVENTS_HEADER

_logger = get_logger("analytics.exporter")

_SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
)


@dataclass(slots=True)
class AnalyticsExporterConfig:
    """Параметры для экспорта аналитики."""

    spreadsheet_id: str
    credentials_path: Path
    events_sheet_name: str
    analytics_sheet_name: str
    flush_interval: int


class AnalyticsExporter:
    """Фоновая задача, синхронизирующая события и агрегаты в Google Sheets."""

    def __init__(self, config: AnalyticsExporterConfig, db_path: Path) -> None:
        self._config = config
        self._db_path = db_path
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._gspread_client: Any | None = None
        self._spreadsheet: Any | None = None
        self._events_ws: Any | None = None
        self._analytics_ws: Any | None = None
        self._lock = asyncio.Lock()
        self._flush_interval = max(config.flush_interval, 5)

    async def start(self) -> None:
        """Запустить фоновую синхронизацию."""

        if self._task is not None:
            return
        if not self._config.spreadsheet_id or not self._config.credentials_path.exists():
            _logger.warning(
                "Аналитика отключена: нет доступа к Google Sheets",
                extra={"stage": "ANALYTICS_DISABLED"},
            )
            return
        try:
            await asyncio.to_thread(self._ensure_setup)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "Не удалось подготовить листы аналитики: %s", exc, extra={"stage": "SETUP_FAILED"}
            )
            return
        self._task = asyncio.create_task(self._run(), name="analytics-exporter")

    async def stop(self) -> None:
        """Остановить экспорт и дождаться завершения фоновой задачи."""

        if self._task is None:
            return
        self._stop_event.set()
        await self._task
        self._task = None
        self._stop_event.clear()

    async def flush_now(self) -> None:
        """Принудительно выполнить выгрузку (используется при остановке)."""

        async with self._lock:
            await self._flush_once()

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._flush_interval)
                except asyncio.TimeoutError:
                    async with self._lock:
                        await self._flush_once()
        finally:
            async with self._lock:
                await self._flush_once()

    async def _flush_once(self) -> None:
        try:
            await asyncio.to_thread(self._ensure_setup)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "Не удалось подготовить листы перед выгрузкой: %s",
                exc,
                extra={"stage": "SETUP_RETRY_FAILED"},
            )
            return

        rows = await track.drain_buffer()
        if not rows:
            # Даже если буфер пуст, всё равно актуализируем метрики.
            await self._update_metrics()
            return
        try:
            await asyncio.to_thread(self._append_events, rows)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "Не удалось записать события в Google Sheets: %s", exc, extra={"stage": "APPEND_FAILED"}
            )
            await track.requeue(rows)
            return
        try:
            await self._update_metrics()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "Не удалось обновить метрики: %s", exc, extra={"stage": "METRICS_FAILED"}
            )

    def _ensure_setup(self) -> None:
        import gspread  # type: ignore[import-not-found]
        from google.oauth2.service_account import Credentials  # type: ignore[import-not-found]

        if self._gspread_client is None:
            credentials = Credentials.from_service_account_file(
                str(self._config.credentials_path), scopes=_SCOPES
            )
            self._gspread_client = gspread.authorize(credentials)
            self._spreadsheet = self._gspread_client.open_by_key(self._config.spreadsheet_id)
        if self._spreadsheet is None:
            raise RuntimeError("Spreadsheet не инициализирован")
        self._events_ws = self._get_or_create_worksheet(
            self._spreadsheet,
            self._config.events_sheet_name,
            EVENTS_HEADER,
        )
        self._analytics_ws = self._get_or_create_worksheet(
            self._spreadsheet,
            self._config.analytics_sheet_name,
            ANALYTICS_HEADER,
        )

    def _get_or_create_worksheet(self, spreadsheet: Any, title: str, header: tuple[str, ...]):
        import gspread  # type: ignore[import-not-found]

        try:
            worksheet = spreadsheet.worksheet(title)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=title, rows=100, cols=len(header))
        self._ensure_header(worksheet, header)
        return worksheet

    def _ensure_header(self, worksheet: Any, header: tuple[str, ...]) -> None:
        first_row = worksheet.row_values(1)
        if list(first_row[: len(header)]) != list(header):
            worksheet.resize(rows=max(worksheet.row_count, 1), cols=len(header))
            worksheet.update(self._range_for_row(1, len(header)), [list(header)])

    def _append_events(self, rows: list[list[str]]) -> None:
        if self._events_ws is None:
            raise RuntimeError("Worksheet 'События' не готов")
        self._with_backoff(self._events_ws.append_rows, rows, value_input_option="USER_ENTERED")

    def _update_metrics_sync(self, metrics_row: list[str]) -> None:
        if self._analytics_ws is None:
            raise RuntimeError("Worksheet 'Аналитика' не готов")
        date_value = metrics_row[0]
        existing_dates = self._analytics_ws.col_values(1)
        try:
            row_index = existing_dates.index(date_value) + 1
        except ValueError:
            row_index = len(existing_dates) + 1
            if row_index <= 1:
                row_index = 2
            if row_index > self._analytics_ws.row_count:
                self._analytics_ws.add_rows(row_index - self._analytics_ws.row_count)
        self._with_backoff(
            self._analytics_ws.update,
            self._range_for_row(row_index, len(metrics_row)),
            [metrics_row],
            value_input_option="USER_ENTERED",
        )

    async def _update_metrics(self) -> None:
        today = date.today()
        metrics_data = await metrics.calculate_daily_metrics(self._db_path, today)
        await asyncio.to_thread(self._update_metrics_sync, metrics_data.to_row())

    @staticmethod
    def _range_for_row(row_index: int, columns: int) -> str:
        return f"A{row_index}:{_column_letter(columns)}{row_index}"

    @staticmethod
    def _with_backoff(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        delay = 1.0
        max_delay = 30.0
        for attempt in range(5):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {429, 500, 503} and attempt < 4:
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    continue
                raise


def _column_letter(index: int) -> str:
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result or "A"


__all__ = ["AnalyticsExporter", "AnalyticsExporterConfig"]
