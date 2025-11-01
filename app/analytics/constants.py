"""Строковые константы для модулей аналитики."""

from __future__ import annotations

from typing import Final

EVENTS_SHEET_DEFAULT: Final[str] = "События"
ANALYTICS_SHEET_DEFAULT: Final[str] = "Аналитика"
DASHBOARD_SHEET_DEFAULT: Final[str] = "Дашборд"

EVENTS_HEADER: Final[tuple[str, ...]] = (
    "Время",
    "Пользователь",
    "Событие",
    "Значение",
    "Доп.данные",
)

ANALYTICS_HEADER: Final[tuple[str, ...]] = (
    "Дата",
    "Пользователи всего",
    "Мужчины",
    "Женщины",
    "% Мужчин",
    "% Женщин",
    "Телефоны (кол-во)",
    "% Телефонов",
    "Возвраты 24ч",
    "Средних примерок на пользователя",
    "Доля завершивших примерку",
    "Клики «Записаться»",
    "CTR «Записаться»",
    "Записей (лидов)",
    "Конверсия в запись",
)

ANALYTICS_COLUMNS: Final[dict[str, str]] = {
    "date": ANALYTICS_HEADER[0],
    "total_users": ANALYTICS_HEADER[1],
    "male_users": ANALYTICS_HEADER[2],
    "female_users": ANALYTICS_HEADER[3],
    "male_percent": ANALYTICS_HEADER[4],
    "female_percent": ANALYTICS_HEADER[5],
    "phones_total": ANALYTICS_HEADER[6],
    "phones_percent": ANALYTICS_HEADER[7],
    "return_visits": ANALYTICS_HEADER[8],
    "avg_generations": ANALYTICS_HEADER[9],
    "finished_share": ANALYTICS_HEADER[10],
    "cta_clicks": ANALYTICS_HEADER[11],
    "cta_ctr": ANALYTICS_HEADER[12],
    "leads": ANALYTICS_HEADER[13],
    "leads_conversion": ANALYTICS_HEADER[14],
}

SHEET_TIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
ANALYTICS_DATE_FORMAT: Final[str] = "%Y-%m-%d"

__all__ = [
    "EVENTS_SHEET_DEFAULT",
    "ANALYTICS_SHEET_DEFAULT",
    "DASHBOARD_SHEET_DEFAULT",
    "EVENTS_HEADER",
    "ANALYTICS_HEADER",
    "ANALYTICS_COLUMNS",
    "SHEET_TIME_FORMAT",
    "ANALYTICS_DATE_FORMAT",
]
