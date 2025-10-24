"""FSM definitions and handler registration."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from aiogram import Router, F
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message
from aiogram.types.input_file import FSInputFile

from app import messages_ru as msg
from app.keyboards import (
    age_keyboard,
    gender_keyboard,
    limit_reached_keyboard,
    models_keyboard,
    result_keyboard,
    retry_keyboard,
    start_keyboard,
    style_keyboard,
)
from app.logging_conf import EVENT_ID
from app.models import FilterOptions, ModelItem
from app.services.catalog_base import CatalogService
from app.services.repository import Repository
from app.services.tryon_base import TryOnService
from app.services.storage_base import StorageService
from app.utils.deeplink import build_ref_link


class TryOnStates(StatesGroup):
    START = State()
    FOR_WHO = State()
    AGE = State()
    STYLE = State()
    WAIT_PHOTO = State()
    SHOW_RECS = State()
    GENERATING = State()
    RESULT = State()
    LIMIT_REACHED = State()
    ERROR = State()


def setup_router(
    repository: Repository,
    catalog: CatalogService,
    tryon: TryOnService,
    storage: StorageService,
    reminder_hours: int,
    bot_username: str,
) -> Router:
    router = Router()
    logger = logging.getLogger("loop_bot.handlers")

    async def _ensure_filters(user_id: int, state: FSMContext) -> FilterOptions:
        data = await state.get_data()
        return FilterOptions(
            gender=data.get("gender", "unisex"),
            age_bucket=data.get("age", "18-24"),
            style=data.get("style", "normal"),
        )

    async def _send_models(message: Message, user_id: int, filters: FilterOptions, state: FSMContext) -> None:
        models = await catalog.pick_four(user_id, filters)
                model_ids: List[str] = []
        for model in models:
            caption = f"{model.meta.title}\nБренд: {model.meta.brand or '—'}"
            try:
                await message.answer_photo(
                    photo=FSInputFile(path=str(model.thumb_path)),
                    caption=caption,
                )
                model_ids.append(model.model_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s Failed to send model %s: %s",
                    EVENT_ID["MODELS_SENT"],
                    model.model_id,
                    exc,
                )
        if model_ids:
            await message.answer(
                "Выберите модель:", reply_markup=models_keyboard(model_ids)
            )
        await state.update_data(current_models=models)
        logger.info("%s Sent %s models", EVENT_ID["MODELS_SENT"], len(models))

    @router.message(CommandStart())
    async def handle_start(message: Message, state: FSMContext) -> None:
        await repository.ensure_user(message.from_user.id)
        text = message.text or ""
        if "ref_" in text:
            parts = text.split()
            if parts and parts[0].startswith("/start") and len(parts) > 1:
                ref_part = parts[1]
                if ref_part.startswith("ref_"):
                    ref_id = ref_part.replace("ref_", "")
                    try:
                        await repository.set_referrer(message.from_user.id, int(ref_id))
                    except ValueError:
                        pass
        await state.set_state(TryOnStates.START)
        await message.answer(msg.WELCOME, reply_markup=start_keyboard())
        logger.info("%s User %s entered start", EVENT_ID["START"], message.from_user.id)

    @router.callback_query(StateFilter(TryOnStates.START), F.data == "start_go")
    async def start_go(callback: CallbackQuery, state: FSMContext) -> None:
        await state.set_state(TryOnStates.FOR_WHO)
        await callback.message.edit_text(msg.FILTER_FOR_WHO, reply_markup=gender_keyboard())
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.START), F.data == "start_info")
    async def start_info(callback: CallbackQuery) -> None:
        await callback.answer(msg.WELCOME_MAGIC, show_alert=True)

    @router.callback_query(StateFilter(TryOnStates.FOR_WHO))
    async def select_gender(callback: CallbackQuery, state: FSMContext) -> None:
        gender = callback.data.replace("gender_", "")
        await repository.update_filters(callback.from_user.id, gender=gender)
        await state.update_data(gender=gender)
        await state.set_state(TryOnStates.AGE)
        await callback.message.edit_text(msg.FILTER_AGE, reply_markup=age_keyboard())
        await callback.answer()
        logger.info("%s Gender selected %s", EVENT_ID["FILTER_SELECTED"], gender)

    @router.callback_query(StateFilter(TryOnStates.AGE))
    async def select_age(callback: CallbackQuery, state: FSMContext) -> None:
        age = callback.data.replace("age_", "")
        await repository.update_filters(callback.from_user.id, age_bucket=age)
        await state.update_data(age=age)
        await state.set_state(TryOnStates.STYLE)
        await callback.message.edit_text(msg.FILTER_STYLE, reply_markup=style_keyboard())
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.STYLE))
    async def select_style(callback: CallbackQuery, state: FSMContext) -> None:
        style = callback.data.replace("style_", "")
        if style == "skip":
            style = "normal"
        await repository.update_filters(callback.from_user.id, style=style)
        await state.update_data(style=style)
        await state.set_state(TryOnStates.WAIT_PHOTO)
        await callback.message.edit_text(msg.ASK_PHOTO)
        await callback.answer()

    @router.message(StateFilter(TryOnStates.WAIT_PHOTO), ~F.photo)
    async def reject_non_photo(message: Message) -> None:
        await message.answer(msg.NOT_PHOTO)

    @router.message(StateFilter(TryOnStates.WAIT_PHOTO), F.photo)
    async def accept_photo(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id
        photo = message.photo[-1]
        filename = f"{photo.file_unique_id}.jpg"
        path = await storage.allocate_upload_path(user_id, filename)
        await photo.download(destination=str(path))
        await state.update_data(upload=str(path))
        await repository.ensure_daily_reset(user_id)
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.LIMIT_REACHED)
            await message.answer(msg.LIMIT_REACHED, reply_markup=limit_reached_keyboard())
            logger.info("%s Limit reached for user %s", EVENT_ID["LIMIT_REACHED"], user_id)
            return
        filters = await _ensure_filters(user_id, state)
        await state.set_state(TryOnStates.SHOW_RECS)
        await message.answer(msg.SHOWING_MODELS)
        await _send_models(message, user_id, filters, state)
        logger.info("%s Photo received from %s", EVENT_ID["PHOTO_RECEIVED"], user_id)

    @router.callback_query(StateFilter(TryOnStates.SHOW_RECS), F.data == "more_models")
    async def more_models(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        filters = await _ensure_filters(user_id, state)
        await _send_models(callback.message, user_id, filters, state)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.SHOW_RECS), F.data.startswith("choose_"))
    async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
        model_id = callback.data.replace("choose_", "")
        data = await state.get_data()
        models_data: List[ModelItem] = data.get("current_models", [])
        selected = next((model for model in models_data if model.model_id == model_id), None)
        if not selected:
            await callback.answer("Модель недоступна", show_alert=True)
            return
        remaining = await repository.remaining_tries(callback.from_user.id)
        if remaining <= 0:
            await state.set_state(TryOnStates.LIMIT_REACHED)
            await callback.message.answer(msg.LIMIT_REACHED, reply_markup=limit_reached_keyboard())
            await callback.answer()
            return
        await state.update_data(selected_model=selected)
        await state.set_state(TryOnStates.GENERATING)
        await callback.message.answer(msg.GENERATING)
        await callback.answer()
        await _perform_generation(callback.message, state, selected)

    async def _perform_generation(message: Message, state: FSMContext, model: ModelItem) -> None:
        user_id = message.chat.id
        data = await state.get_data()
        upload_path = data.get("upload")
        if not upload_path:
            await message.answer(msg.GENERATION_FAILED, reply_markup=retry_keyboard())
            await state.set_state(TryOnStates.ERROR)
            return
        session_id = uuid.uuid4().hex
        try:
            results = await tryon.generate(
                user_id=user_id,
                session_id=session_id,
                input_photo=Path(upload_path),
                overlays=[model.overlay_path],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s Generation failed: %s", EVENT_ID["GENERATION_FAILED"], exc
            )
            await message.answer(msg.GENERATION_FAILED, reply_markup=retry_keyboard())
            await state.set_state(TryOnStates.ERROR)
            return
        await repository.inc_used_on_success(user_id)
        await state.set_state(TryOnStates.RESULT)
        await message.answer(msg.GENERATION_SUCCESS)
        for result_path in results:
            await message.answer_photo(FSInputFile(path=str(result_path)))
        ref_link = build_ref_link(bot_username, user_id)
        await message.answer(
            "Поделиться ссылкой: " + ref_link,
            reply_markup=result_keyboard(model.meta.product_url, ref_link),
        )
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.LIMIT_REACHED)
            await message.answer(msg.LIMIT_REACHED, reply_markup=limit_reached_keyboard())
            logger.info("%s Limit reached post generation %s", EVENT_ID["LIMIT_REACHED"], user_id)
        logger.info("%s Generation succeeded for %s", EVENT_ID["GENERATION_SUCCESS"], user_id)

    @router.callback_query(StateFilter(TryOnStates.RESULT), F.data == "result_more")
    async def result_more(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        filters = await _ensure_filters(user_id, state)
        await state.set_state(TryOnStates.SHOW_RECS)
        await callback.message.answer(msg.SHOWING_MODELS)
        await _send_models(callback.message, user_id, filters, state)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.LIMIT_REACHED), F.data == "limit_promo")
    async def limit_promo(callback: CallbackQuery) -> None:
        await callback.message.answer(msg.PROMO_CODE)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.LIMIT_REACHED), F.data == "limit_remind")
    async def limit_remind(callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        when = datetime.now(UTC) + timedelta(hours=reminder_hours)
        await repository.set_reminder(user_id, when)
        await callback.message.answer("Напомню через сутки!")
        await callback.answer()
        logger.info(
            "%s Reminder scheduled for %s", EVENT_ID["REMINDER_SCHEDULED"], user_id
        )

    @router.callback_query(StateFilter(TryOnStates.ERROR), F.data == "retry")
    async def retry(callback: CallbackQuery, state: FSMContext) -> None:
        data = await state.get_data()
        selected: Optional[ModelItem] = data.get("selected_model")
        if not selected:
            await callback.answer("Нет выбранной модели", show_alert=True)
            return
        model = selected
        await state.set_state(TryOnStates.GENERATING)
        await callback.message.answer(msg.GENERATING)
        await callback.answer()
        await _perform_generation(callback.message, state, model)

    return router
