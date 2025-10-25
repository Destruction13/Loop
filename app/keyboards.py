"""Inline keyboards used across the bot."""

from __future__ import annotations

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


def model_card_keyboard(unique_id: str, title: str, site_url: str) -> InlineKeyboardMarkup:
    """Keyboard attached to a catalog card."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"Выбрать «{title}»", callback_data=f"pick|{unique_id}")],
            [InlineKeyboardButton(text="Подробнее о модели", url=site_url)],
        ]
    )


def more_models_keyboard() -> InlineKeyboardMarkup:
    """Keyboard offering to fetch more models."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Ещё 4", callback_data="more_models")]
        ]
    )


def result_keyboard(product_url: str, ref_url: str) -> InlineKeyboardMarkup:
    """Keyboard for the result screen."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Ещё", callback_data="result_more")],
            [InlineKeyboardButton(text="Посмотреть/Купить", url=product_url)],
            [InlineKeyboardButton(text="Поделиться", url=ref_url)],
        ]
    )


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
