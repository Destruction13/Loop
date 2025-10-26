from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

import messages
from bot import keyboards
from bot.idle import IdleWatcher
from bot.states import TryOnStates
from db.init import Database

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StartConfig:
    promo_video_1: Path | None
    promo_video_2: Path | None


def register_start_handlers(router: Router, db: Database, idle: IdleWatcher, config: StartConfig) -> None:
    @router.message(CommandStart())
    async def handle_start(message: Message, state: FSMContext) -> None:
        tg_id = message.from_user.id
        await idle.touch(tg_id)
        await db.upsert_user(tg_id, message.from_user.username, message.from_user.full_name)
        await db.log_event(
            tg_id,
            "start",
            {
                "username": message.from_user.username,
                "name": message.from_user.full_name,
            },
        )
        await state.clear()

        await _send_video_if_exists(message, config.promo_video_1, messages.PROMO_GREETING)
        await _send_video_if_exists(message, config.promo_video_2, messages.PROMO_INSTRUCTION)

        await message.answer(messages.GENDER_PROMPT, reply_markup=keyboards.gender_keyboard())
        await state.set_state(TryOnStates.waiting_for_gender)

    @router.callback_query(keyboards.GenderCallback.filter())
    async def handle_gender(call: CallbackQuery, callback_data: keyboards.GenderCallback, state: FSMContext) -> None:
        await call.answer()
        if call.message is None:
            return
        tg_id = call.from_user.id
        await idle.touch(tg_id)
        gender = callback_data.value
        if gender not in {"male", "female"}:
            return
        await db.update_gender(tg_id, gender)
        await db.log_event(tg_id, "gender_selected", {"gender": gender})
        await call.message.edit_reply_markup()
        await call.message.answer(messages.PHOTO_PROMPT)
        await state.set_state(TryOnStates.waiting_for_photo)


async def _send_video_if_exists(message: Message, path: Path | None, caption: str) -> None:
    if path is None:
        return
    if not path.exists():
        LOGGER.warning("Promo video %s is missing", path)
        return
    try:
        await message.answer_video(video=path.read_bytes(), caption=caption)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to send promo video %s: %s", path, exc)
        await message.answer(caption)
