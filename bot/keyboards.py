from __future__ import annotations

from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

import messages


class GenderCallback(CallbackData, prefix="gender"):
    value: str


class ModelSelectCallback(CallbackData, prefix="model"):
    model_id: str


class MoreOptionsCallback(CallbackData, prefix="more"):
    action: str


def gender_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=messages.GENDER_BUTTON_MALE, callback_data=GenderCallback(value="male").pack())],
            [InlineKeyboardButton(text=messages.GENDER_BUTTON_FEMALE, callback_data=GenderCallback(value="female").pack())],
        ]
    )


def request_contact_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=messages.CONTACT_BUTTON_TEXT, request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
        selective=True,
    )


def remove_reply_keyboard() -> ReplyKeyboardRemove:
    return ReplyKeyboardRemove()


def model_choice_keyboard(models: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    inline_keyboard = [
        [InlineKeyboardButton(text=title, callback_data=ModelSelectCallback(model_id=model_id).pack())]
        for model_id, title in models
    ]
    inline_keyboard.append(
        [
            InlineKeyboardButton(
                text=messages.MORE_VARIANTS_BUTTON_TEXT,
                callback_data=MoreOptionsCallback(action="more").pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=inline_keyboard, row_width=1)


def selected_model_keyboard(product_url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=messages.DETAILS_BUTTON_TEXT, url=product_url)],
            [
                InlineKeyboardButton(
                    text=messages.MORE_VARIANTS_BUTTON_TEXT,
                    callback_data=MoreOptionsCallback(action="more").pack(),
                )
            ],
        ],
        row_width=1,
    )


def ecommerce_cta_keyboard(ecom_button_text: str, ecom_url: str, more_button_text: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=ecom_button_text, url=ecom_url)],
            [InlineKeyboardButton(text=more_button_text, callback_data=MoreOptionsCallback(action="more").pack())],
        ],
        row_width=1,
    )


def socials_keyboard(buttons: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    inline_keyboard = [[InlineKeyboardButton(text=text, url=url)] for text, url in buttons]
    return InlineKeyboardMarkup(inline_keyboard=inline_keyboard, row_width=1)
