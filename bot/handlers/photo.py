from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, Message

import messages
from bot import keyboards
from bot.generator import TryOnGenerator
from bot.idle import IdleWatcher
from bot.states import TryOnStates
from db.init import Database


@dataclass(slots=True)
class PhotoHandlerConfig:
    uploads_dir: Path
    phone_request_after: int


def register_photo_handlers(
    router: Router,
    db: Database,
    generator: TryOnGenerator,
    idle: IdleWatcher,
    config: PhotoHandlerConfig,
) -> None:
    config.uploads_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    @router.message(TryOnStates.waiting_for_photo, F.photo)
    async def handle_photo(message: Message, state: FSMContext) -> None:
        tg_id = message.from_user.id
        await idle.touch(tg_id)
        user = await db.get_user(tg_id)
        if user is None or not user.get("gender"):
            await message.answer(messages.GENDER_PROMPT, reply_markup=keyboards.gender_keyboard())
            await state.set_state(TryOnStates.waiting_for_gender)
            return

        file_unique = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        photo_path = config.uploads_dir / f"{tg_id}_{file_unique}.jpg"
        photo = message.photo[-1]
        await photo.download(destination=photo_path)
        await db.log_event(
            tg_id,
            "photo_received",
            {"file_id": photo.file_id, "path": str(photo_path)},
        )

        attempt = await db.increment_attempt_count(tg_id)
        try:
            result = await generator.generate(user["gender"] or "unisex")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate collage for %s: %s", tg_id, exc)
            await message.answer(messages.RETRY_LATER_TEXT)
            return

        caption = random.choice(list(messages.GENERATION_CAPTIONS))
        collage_input = BufferedInputFile(result.image.getvalue(), filename=f"collage_{file_unique}.jpg")
        collage_message = await message.answer_photo(
            collage_input,
            caption=caption,
            reply_markup=keyboards.model_choice_keyboard(
                [(model.model_id, model.title) for model in result.models]
            ),
        )

        await db.log_event(
            tg_id,
            "collage_generated",
            {
                "models": [model.model_id for model in result.models],
                "caption": caption,
            },
        )
        for model in result.models:
            await db.log_event(
                tg_id,
                "model_viewed",
                {"model_id": model.model_id, "title": model.title},
            )

        await state.update_data(
            models=[model.__dict__ for model in result.models],
            collage_message_id=collage_message.message_id,
        )

        await message.answer(random.choice(list(messages.TIP_CAPTIONS)))
        await state.set_state(TryOnStates.waiting_for_photo)

        user_record = await db.get_user(tg_id)
        if (
            user_record
            and not user_record.get("phone")
            and config.phone_request_after > 0
            and attempt == config.phone_request_after
        ):
            await message.answer(
                messages.CONTACT_REQUEST_TEXT,
                reply_markup=keyboards.request_contact_keyboard(),
            )

    @router.message(~StateFilter(TryOnStates.waiting_for_photo), F.photo)
    async def handle_photo_wrong_state(message: Message) -> None:
        await idle.touch(message.from_user.id)
        await message.answer(messages.NEW_PHOTO_REQUIRED_TEXT)

    @router.message(F.text, ~F.text.startswith("/"))
    async def handle_text(message: Message, state: FSMContext) -> None:
        await idle.touch(message.from_user.id)
        current_state = await state.get_state()
        if current_state == TryOnStates.waiting_for_photo.state:
            await message.answer(messages.PHOTO_PROMPT)
