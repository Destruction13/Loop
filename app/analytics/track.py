"""Запись событий аналитики в SQLite и буфер для экспорта."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Sequence

from logger import get_logger

from .constants import SHEET_TIME_FORMAT

_logger = get_logger("analytics.track")


@dataclass(slots=True)
class _SheetRow:
    """Описывает строку, которую нужно выгрузить в Google Sheets."""

    timestamp: datetime
    user_id: str
    event: str
    value: str | None
    meta_json: str | None

    def to_values(self) -> list[str]:
        """Вернуть список значений для append_rows."""

        local_time = self.timestamp.astimezone().strftime(SHEET_TIME_FORMAT)
        return [
            local_time,
            self.user_id,
            self.event,
            self.value or "",
            self.meta_json or "",
        ]


_db_path: Path | None = None
_buffer: Deque[_SheetRow] = deque()
_buffer_lock = asyncio.Lock()
_init_lock = asyncio.Lock()
_initialized = False

_RETURN_CHECK_EVENTS = {"start", "photo_uploaded"}


def _ensure_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("analytics.track.init() не вызван")
    return _db_path


def _create_schema(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event TEXT NOT NULL,
                value TEXT,
                meta_json TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analytics_user_ts ON analytics_events(user_id, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analytics_event_ts ON analytics_events(event, ts)"
        )
        conn.commit()
    finally:
        conn.close()


async def init(db_path: Path) -> None:
    """Инициализировать подсистему аналитики и создать схему при необходимости."""

    global _initialized, _db_path
    async with _init_lock:
        if _initialized:
            return
        _db_path = db_path
        await asyncio.to_thread(_create_schema, db_path)
        _initialized = True
        _logger.debug("Схема analytics_events готова", extra={"stage": "ANALYTICS_DB_READY"})


def _insert_event(path: Path, row: _SheetRow) -> None:
    payload = (
        row.user_id,
        row.event,
        row.value,
        row.meta_json,
        row.timestamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
    )
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO analytics_events (user_id, event, value, meta_json, ts) VALUES (?, ?, ?, ?, ?)",
            payload,
        )
        conn.commit()
    finally:
        conn.close()


async def _push_to_buffer(row: _SheetRow) -> None:
    async with _buffer_lock:
        _buffer.append(row)


async def drain_buffer() -> list[list[str]]:
    """Забрать все накопленные строки для выгрузки и очистить буфер."""

    async with _buffer_lock:
        rows = list(_buffer)
        _buffer.clear()
    return [item.to_values() for item in rows]


async def requeue(rows: Sequence[list[str]]) -> None:
    """Вернуть строки обратно в буфер (используется при ошибках выгрузки)."""

    if not rows:
        return
    async with _buffer_lock:
        for row in reversed(rows):
            timestamp_str, user_id, event, value, meta_json = (row + [None] * 5)[:5]
            try:
                timestamp = datetime.strptime(timestamp_str, SHEET_TIME_FORMAT)
                timestamp = timestamp.replace(tzinfo=datetime.now().astimezone().tzinfo)
            except Exception:  # noqa: BLE001 - защитный сценарий
                timestamp = datetime.now(timezone.utc)
            buffer_row = _SheetRow(
                timestamp=timestamp.astimezone(timezone.utc),
                user_id=str(user_id or ""),
                event=str(event or ""),
                value=str(value) if value is not None else None,
                meta_json=str(meta_json) if meta_json is not None else None,
            )
            _buffer.appendleft(buffer_row)


async def _store_event(row: _SheetRow) -> None:
    path = _ensure_db_path()
    await asyncio.to_thread(_insert_event, path, row)
    await _push_to_buffer(row)


def _load_last_timestamp(path: Path, user_id: str) -> datetime | None:
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT ts FROM analytics_events WHERE user_id = ? ORDER BY ts DESC LIMIT 1",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        ts_raw = row["ts"]
        try:
            return datetime.fromisoformat(str(ts_raw))
        except ValueError:
            return None
    finally:
        conn.close()


async def track_event(
    user_id: str | int,
    event: str,
    value: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Записать событие пользователя в базу и добавить в буфер."""

    if not _initialized:
        raise RuntimeError("analytics.track.init() не вызван")
    user_text = str(user_id)
    now_utc = datetime.now(timezone.utc)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta else None
    row = _SheetRow(
        timestamp=now_utc,
        user_id=user_text,
        event=event,
        value=value,
        meta_json=meta_json,
    )

    path = _ensure_db_path()
    should_emit_return = False
    if event in _RETURN_CHECK_EVENTS:
        last_ts = await asyncio.to_thread(_load_last_timestamp, path, user_text)
        if last_ts is not None:
            last_ts_utc = last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=timezone.utc)
            if now_utc - last_ts_utc >= timedelta(hours=24):
                should_emit_return = True
    await _store_event(row)
    if should_emit_return:
        await _store_event(
            _SheetRow(
                timestamp=now_utc,
                user_id=user_text,
                event="return_visit_24h",
                value=None,
                meta_json=json.dumps({"source": event}, ensure_ascii=False, sort_keys=True),
            )
        )


def get_db_path() -> Path:
    """Вернуть путь до базы для расчёта метрик."""

    return _ensure_db_path()


__all__ = ["init", "track_event", "drain_buffer", "requeue", "get_db_path"]
