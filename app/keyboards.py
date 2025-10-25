"""Inline keyboards used across the bot."""

from __future__ import annotations

from typing import Sequence

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def start_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for the start menu."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Поехали", callback_data="start_go")],
            [InlineKeyboardButton(text="Что за магия?", callback_data="start_info")],
        ]
    )


def gender_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for selecting gender."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Мужчина", callback_data="gender_male")],
            [InlineKeyboardButton(text="Женщина", callback_data="gender_female")],
            [InlineKeyboardButton(text="Унисекс", callback_data="gender_unisex")],
        ]
    )


def age_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for selecting age bucket."""

    ages = [
        ("13–17", "age_13-17"),
        ("18–24", "age_18-24"),
        ("25–34", "age_25-34"),
        ("35–44", "age_35-44"),
        ("45–55", "age_45-55"),
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=data)] for label, data in ages
        ]
    )


def style_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for selecting style preference."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Обычный", callback_data="style_normal")],
            [InlineKeyboardButton(text="Необычный", callback_data="style_bold")],
            [InlineKeyboardButton(text="Пропустить", callback_data="style_skip")],
        ]
    )


def pair_selection_keyboard(
    models: Sequence[tuple[str, str]],
    *,
    max_title_length: int = 28,
) -> InlineKeyboardMarkup:
    """Keyboard for selecting one of the models in a pair."""

    if not models:
        return InlineKeyboardMarkup(inline_keyboard=[])

    buttons: list[InlineKeyboardButton] = []
    for unique_id, title in models:
        truncated = _truncate_title(title, max_title_length)
        buttons.append(
            InlineKeyboardButton(
                text=truncated,
                callback_data=f"pick|{unique_id}",
            )
        )
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def _truncate_title(title: str, max_length: int) -> str:
    if len(title) <= max_length:
        return title
    if max_length <= 1:
        return "…"
    return title[: max_length - 1] + "…"


def generation_result_keyboard(site_url: str, remaining: int) -> InlineKeyboardMarkup:
    """Keyboard attached to the generation result message."""

    buttons = [InlineKeyboardButton(text="Подробнее о модели", url=site_url)]
    if remaining > 0:
        buttons.append(
            InlineKeyboardButton(
                text=f"Ещё варианты (осталось {remaining})",
                callback_data=f"more|{remaining}",
            )
        )
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def limit_reached_keyboard() -> InlineKeyboardMarkup:
    """Keyboard displayed when daily limit is reached."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Получить промокод", callback_data="limit_promo")],
            [InlineKeyboardButton(text="Записаться", url="https://example.com/booking")],
            [InlineKeyboardButton(text="Вернуться завтра", callback_data="limit_remind")],
        ]
    )


def retry_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for retrying failed generation."""

    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Повторить генерацию", callback_data="retry")]]
    )
