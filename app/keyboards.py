"""Inline keyboards used across the bot."""

from __future__ import annotations

from typing import Sequence

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.texts import messages as msg


def start_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for the start menu."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.START_PRIMARY_BUTTON,
                    callback_data="start_go",
                )
            ],
            [
                InlineKeyboardButton(
                    text=msg.START_INFO_BUTTON,
                    callback_data="start_info",
                )
            ],
        ]
    )


def gender_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for selecting gender."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.GENDER_BUTTON_MALE,
                    callback_data="gender_male",
                )
            ],
            [
                InlineKeyboardButton(
                    text=msg.GENDER_BUTTON_FEMALE,
                    callback_data="gender_female",
                )
            ],
            [
                InlineKeyboardButton(
                    text=msg.GENDER_BUTTON_UNISEX,
                    callback_data="gender_unisex",
                )
            ],
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
                callback_data=f"pick:{unique_id}",
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

    buttons = [InlineKeyboardButton(text=msg.DETAILS_BUTTON_TEXT, url=site_url)]
    if remaining > 0:
        buttons.append(
            InlineKeyboardButton(
                text=f"{msg.MORE_VARIANTS_BUTTON_TEXT} (осталось {remaining})",
                callback_data=f"more|{remaining}",
            )
        )
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def limit_reached_keyboard(landing_url: str) -> InlineKeyboardMarkup:
    """Keyboard displayed when daily limit is reached."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=landing_url)],
            [
                InlineKeyboardButton(
                    text=msg.PROMO_BUTTON_TEXT,
                    callback_data="limit_promo",
                )
            ],
            [
                InlineKeyboardButton(
                    text=msg.REMIND_LATER_BUTTON_TEXT,
                    callback_data="limit_remind",
                )
            ],
        ]
    )


def promo_keyboard(landing_url: str) -> InlineKeyboardMarkup:
    """Keyboard attached to the promo code message."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=landing_url)],
            [
                InlineKeyboardButton(
                    text=msg.REMIND_LATER_BUTTON_TEXT,
                    callback_data="limit_remind",
                )
            ],
        ]
    )


def retry_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for retrying failed generation."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.RETRY_GENERATION_BUTTON_TEXT,
                    callback_data="retry",
                )
            ]
        ]
    )


def reminder_keyboard() -> InlineKeyboardMarkup:
    """Keyboard attached to reminder messages."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.REMINDER_BUTTON_TEXT,
                    callback_data="reminder_go",
                )
            ]
        ]
    )


def all_seen_keyboard(landing_url: str) -> InlineKeyboardMarkup:
    """Keyboard shown when подборки закончились."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.REMIND_LATER_BUTTON_TEXT,
                    callback_data="limit_remind",
                ),
                InlineKeyboardButton(
                    text=msg.BOOKING_BUTTON_TEXT,
                    url=landing_url,
                ),
            ]
        ]
    )
