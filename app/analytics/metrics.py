"""Расчёт KPI по событиям из таблицы analytics_events."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

from logger import get_logger

from .constants import ANALYTICS_COLUMNS

_logger = get_logger("analytics.metrics")


@dataclass(slots=True)
class DailyMetrics:
    """Набор метрик за сутки.

    Формулы (все проценты считаются от общего числа уникальных пользователей за сутки):
    * «Пользователи всего» — количество уникальных user_id за период.
    * «Мужчины»/«Женщины» — по последнему выбранному гендеру пользователя в периоде.
      Проценты — доля от «Пользователи всего».
    * «Телефоны (кол-во)» — число уникальных пользователей, отправивших телефон.
      Процент — доля от «Пользователи всего».
    * «Возвраты 24ч» — количество событий return_visit_24h.
    * «Средних примерок на пользователя» — total(generation_started) / «Пользователи всего».
    * «Доля завершивших примерку» — доля пользователей, у которых было ≥1 generation_finished.
    * «Клики «Записаться»» — число уникальных пользователей с событием cta_book_opened.
      CTR — доля от «Пользователи всего».
    * «Записей (лидов)» — число уникальных пользователей с lead_export_ok.
      Конверсия — доля от «Пользователи всего».
    """

    day: date
    total_users: int
    male_users: int
    female_users: int
    male_percent: float
    female_percent: float
    phones_total: int
    phones_percent: float
    return_visits: int
    avg_generations: float
    finished_share: float
    cta_clicks: int
    cta_ctr: float
    leads: int
    leads_conversion: float

    def to_row(self) -> list[str]:
        """Сконвертировать метрики в строку для листа «Аналитика»."""

        return [
            self.day.isoformat(),
            str(self.total_users),
            str(self.male_users),
            str(self.female_users),
            f"{self.male_percent:.2f}",
            f"{self.female_percent:.2f}",
            str(self.phones_total),
            f"{self.phones_percent:.2f}",
            str(self.return_visits),
            f"{self.avg_generations:.2f}",
            f"{self.finished_share:.2f}",
            str(self.cta_clicks),
            f"{self.cta_ctr:.2f}",
            str(self.leads),
            f"{self.leads_conversion:.2f}",
        ]


def _local_bounds(target_day: date) -> tuple[datetime, datetime]:
    local_now = datetime.now().astimezone()
    tz = local_now.tzinfo or timezone.utc
    start_local = datetime.combine(target_day, dt_time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _fetch_rows(db_path: Path, start_iso: str, end_iso: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT user_id, event, value, ts
            FROM analytics_events
            WHERE ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            (start_iso, end_iso),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def _calculate_metrics_sync(db_path: Path, target_day: date) -> DailyMetrics:
    start_utc, end_utc = _local_bounds(target_day)
    rows = _fetch_rows(db_path, start_utc.isoformat(timespec="seconds"), end_utc.isoformat(timespec="seconds"))

    total_users: set[str] = set()
    phones_users: set[str] = set()
    cta_users: set[str] = set()
    lead_users: set[str] = set()
    finished_users: set[str] = set()
    gender_latest: dict[str, tuple[datetime, str]] = {}
    gen_started_counts: dict[str, int] = {}
    return_visits = 0

    for row in rows:
        user_id = str(row["user_id"])
        event = str(row["event"])
        value = row["value"]
        ts_raw = str(row["ts"])
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            ts = start_utc
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        total_users.add(user_id)

        if event == "gender_selected":
            gender_latest[user_id] = (ts, str(value or ""))
        elif event == "phone_shared":
            phones_users.add(user_id)
        elif event == "generation_started":
            gen_started_counts[user_id] = gen_started_counts.get(user_id, 0) + 1
        elif event == "generation_finished":
            finished_users.add(user_id)
        elif event == "cta_book_opened":
            cta_users.add(user_id)
        elif event == "lead_export_ok":
            lead_users.add(user_id)
        elif event == "return_visit_24h":
            return_visits += 1

    male_users = 0
    female_users = 0
    for user_id, (ts, gender_value) in gender_latest.items():
        if gender_value == "male":
            male_users += 1
        elif gender_value == "female":
            female_users += 1
        else:
            # Пользователь выбрал другой вариант — игнорируем в разделении по полу.
            _logger.debug(
                "Неизвестный гендер %s для пользователя %s", gender_value, user_id
            )

    total_count = len(total_users)
    male_percent = (male_users / total_count * 100) if total_count else 0.0
    female_percent = (female_users / total_count * 100) if total_count else 0.0
    phones_percent = (len(phones_users) / total_count * 100) if total_count else 0.0
    avg_generations = (
        sum(gen_started_counts.values()) / total_count if total_count else 0.0
    )
    finished_share = (len(finished_users) / total_count * 100) if total_count else 0.0
    cta_ctr = (len(cta_users) / total_count * 100) if total_count else 0.0
    leads_conversion = (len(lead_users) / total_count * 100) if total_count else 0.0

    return DailyMetrics(
        day=target_day,
        total_users=total_count,
        male_users=male_users,
        female_users=female_users,
        male_percent=male_percent,
        female_percent=female_percent,
        phones_total=len(phones_users),
        phones_percent=phones_percent,
        return_visits=return_visits,
        avg_generations=avg_generations,
        finished_share=finished_share,
        cta_clicks=len(cta_users),
        cta_ctr=cta_ctr,
        leads=len(lead_users),
        leads_conversion=leads_conversion,
    )


async def calculate_daily_metrics(db_path: Path, target_day: date) -> DailyMetrics:
    """Асинхронная обёртка вокруг синхронного расчёта метрик."""

    return await asyncio.to_thread(_calculate_metrics_sync, db_path, target_day)


__all__ = ["DailyMetrics", "calculate_daily_metrics", "ANALYTICS_COLUMNS"]
