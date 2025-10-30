"""Inline keyboards used across the bot."""

from __future__ import annotations

from typing import Sequence

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from aiogram.types.web_app_info import WebAppInfo

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


def main_reply_keyboard(policy_url: str | None = None) -> ReplyKeyboardMarkup:
    """Persistent reply keyboard with quick actions."""

    sanitized = (policy_url or "").strip()
    if sanitized:
        policy_button = KeyboardButton(
            text=msg.MAIN_MENU_POLICY_BUTTON,
            web_app=WebAppInfo(url=sanitized),
        )
    else:
        policy_button = KeyboardButton(text=msg.MAIN_MENU_POLICY_BUTTON)

    return ReplyKeyboardMarkup(
        resize_keyboard=True,
        one_time_keyboard=False,
        keyboard=[
            [KeyboardButton(text=msg.MAIN_MENU_TRY_BUTTON)],
            [policy_button],
        ],
    )


def privacy_policy_keyboard(url: str) -> InlineKeyboardMarkup | None:
    """Inline keyboard linking to the privacy policy URL."""

    sanitized = (url or "").strip()
    if not sanitized:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=msg.MAIN_MENU_POLICY_BUTTON, url=sanitized)]
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
        ]
    )


def batch_selection_keyboard(
    models: Sequence[tuple[str, str]],
    *,
    source: str,
    max_title_length: int = 28,
) -> InlineKeyboardMarkup:
    """Keyboard for selecting one of the models in a batch."""

    if not models:
        return InlineKeyboardMarkup(inline_keyboard=[])

    buttons: list[InlineKeyboardButton] = []
    for unique_id, title in models:
        truncated = _truncate_title(title, max_title_length)
        buttons.append(
            InlineKeyboardButton(
                text=truncated,
                callback_data=f"pick:{source}:{unique_id}",
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

    rows = [[InlineKeyboardButton(text=msg.DETAILS_BUTTON_TEXT, url=site_url)]]
    if remaining > 0:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{msg.MORE_VARIANTS_BUTTON_TEXT} (осталось {remaining})",
                    callback_data=f"more|{remaining}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def limit_reached_keyboard(site_url: str) -> InlineKeyboardMarkup:
    """Keyboard displayed when daily limit is reached."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=site_url)],
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


def promo_keyboard(site_url: str) -> InlineKeyboardMarkup:
    """Keyboard attached to the promo code message."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=site_url)],
            [
                InlineKeyboardButton(
                    text=msg.REMIND_LATER_BUTTON_TEXT,
                    callback_data="limit_remind",
                )
            ],
        ]
    )


def send_new_photo_keyboard() -> InlineKeyboardMarkup:
    """Keyboard prompting the user to provide a different photo."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.BTN_SEND_NEW_PHOTO,
                    callback_data="send_new_photo",
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


def idle_reminder_keyboard(site_url: str) -> InlineKeyboardMarkup:
    """Keyboard shown in idle reminder messages."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.MORE_VARIANTS_BUTTON_TEXT,
                    callback_data="more|idle",
                )
            ],
            [
                InlineKeyboardButton(
                    text=msg.IDLE_REMINDER_BUTTON_GO_SITE,
                    url=site_url,
                )
            ],
        ]
    )


def social_ad_keyboard(
    social_links: Sequence[tuple[str, str]]
) -> InlineKeyboardMarkup:
    """Keyboard shown in social media advertisement messages."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=title, url=url)]
            for title, url in social_links
        ]
        + [
            [
                InlineKeyboardButton(
                    text=msg.SOCIAL_AD_BTN_TRY,
                    callback_data="more|social",
                )
            ],
        ]
    )


def all_seen_keyboard(site_url: str) -> InlineKeyboardMarkup:
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
                    url=site_url,
                ),
            ]
        ]
    )


def contact_request_keyboard() -> ReplyKeyboardMarkup:
    """Reply keyboard prompting the user to share their phone number."""

    return ReplyKeyboardMarkup(
        resize_keyboard=True,
        one_time_keyboard=False,
        keyboard=[
            [
                KeyboardButton(
                    text=msg.ASK_PHONE_BUTTON_SHARE,
                    request_contact=True,
                )
            ],
            [
                KeyboardButton(text=msg.ASK_PHONE_BUTTON_SKIP),
                KeyboardButton(text=msg.ASK_PHONE_BUTTON_NEVER),
            ],
        ],
    )
