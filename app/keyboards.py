"""Inline keyboards used across the bot."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.texts import messages as msg


def _booking_button(site_url: str, *, as_callback: bool) -> InlineKeyboardButton:
    sanitized = (site_url or "").strip()
    if as_callback or not sanitized:
        return InlineKeyboardButton(
            text=msg.BOOKING_BUTTON_TEXT,
            callback_data="cta_book",
        )
    return InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=sanitized)


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


def limit_reached_keyboard(site_url: str, *, use_callback: bool = False) -> InlineKeyboardMarkup:
    """Keyboard displayed when daily limit is reached."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [_booking_button(site_url, as_callback=use_callback)],
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


def promo_keyboard(site_url: str, *, use_callback: bool = False) -> InlineKeyboardMarkup:
    """Keyboard attached to the promo code message."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [_booking_button(site_url, as_callback=use_callback)],
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


def social_ad_keyboard(links: Sequence[tuple[str, str]]) -> InlineKeyboardMarkup:
    """Keyboard shown in social media advertisement messages."""

    rows: list[list[InlineKeyboardButton]] = []
    for title, url in links:
        title_clean = title.strip()
        url_clean = url.strip()
        if not title_clean or not url_clean:
            continue
        rows.append([InlineKeyboardButton(text=title_clean, url=url_clean)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def all_seen_keyboard(site_url: str) -> InlineKeyboardMarkup:
    """Keyboard shown when подборки закончились."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.REMIND_LATER_BUTTON_TEXT,
                    callback_data="limit_remind",
                ),
                _booking_button(site_url, as_callback=False),
            ]
        ]
    )


def remove_more_button(markup: InlineKeyboardMarkup | None) -> InlineKeyboardMarkup | None:
    """Return a copy of markup without buttons invoking more callbacks."""

    if not markup or not markup.inline_keyboard:
        return None
    new_rows = []
    changed = False
    for row in markup.inline_keyboard:
        new_row = []
        for button in row:
            callback_data = getattr(button, "callback_data", None)
            if callback_data and callback_data.startswith("more|"):
                changed = True
                continue
            new_row.append(button)
        if new_row:
            new_rows.append(new_row)
    if not changed:
        return None
    return InlineKeyboardMarkup(inline_keyboard=new_rows)


def more_buttonless_markup(
    message_type: str, payload: Mapping[str, Any] | None = None
) -> InlineKeyboardMarkup | None:
    """Construct markup for supported message types without the more button."""

    payload_data: Mapping[str, Any] = payload or {}
    if message_type == "result":
        site_url = payload_data.get("site_url")
        if not site_url:
            return None
        return generation_result_keyboard(str(site_url), 0)
    if message_type == "idle":
        site_url = payload_data.get("site_url")
        if not site_url:
            return None
        trimmed = remove_more_button(idle_reminder_keyboard(str(site_url)))
        if trimmed is not None:
            return trimmed
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=msg.IDLE_REMINDER_BUTTON_GO_SITE,
                        url=str(site_url),
                    )
                ]
            ]
        )
    if message_type == "social":
        links_raw = payload_data.get("links")
        links: list[tuple[str, str]] = []
        if isinstance(links_raw, Sequence) and not isinstance(links_raw, (str, bytes)):
            for entry in links_raw:
                if not isinstance(entry, Mapping):
                    continue
                title = str(entry.get("title") or "").strip()
                url = str(entry.get("url") or "").strip()
                if title and url:
                    links.append((title, url))
        if not links:
            instagram_url = str(payload_data.get("instagram_url") or "").strip()
            if instagram_url:
                links.append(("Instagram", instagram_url))
        if not links:
            return None
        return social_ad_keyboard(links)
    return None
CONTACT_SHARE_CALLBACK = "contact_share"
CONTACT_SKIP_CALLBACK = "contact_skip"
CONTACT_NEVER_CALLBACK = "contact_never"


def contact_request_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard prompting the user to share their phone number."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=msg.ASK_PHONE_BUTTON_SHARE,
                    callback_data=CONTACT_SHARE_CALLBACK,
                )
            ],
            [
                InlineKeyboardButton(
                    text=msg.ASK_PHONE_BUTTON_SKIP,
                    callback_data=CONTACT_SKIP_CALLBACK,
                ),
                InlineKeyboardButton(
                    text=msg.ASK_PHONE_BUTTON_NEVER,
                    callback_data=CONTACT_NEVER_CALLBACK,
                ),
            ],
        ]
    )
