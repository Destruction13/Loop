"""FSM definitions and handler registration."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from aiogram import Router, F
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message
from aiogram.types.input_file import (
    BufferedInputFile,
    FSInputFile,
    URLInputFile,
)

from app import messages_ru as msg
from app.keyboards import (
    age_keyboard,
    gender_keyboard,
    generation_result_keyboard,
    limit_reached_keyboard,
    pair_selection_keyboard,
    retry_keyboard,
    start_keyboard,
    style_keyboard,
)
from app.logging_conf import EVENT_ID
from app.models import FilterOptions, GlassModel
from app.services.catalog_base import CatalogError, CatalogService
from app.services.collage import CollageResult, CollageService
from app.services.repository import Repository
from app.services.tryon_base import TryOnService
from app.services.storage_base import StorageService


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
    collage: CollageService,
    reminder_hours: int,
) -> Router:
    router = Router()
    logger = logging.getLogger("loop_bot.handlers")

    def _catalog_gender(value: str) -> str:
        mapping = {
            "male": "Мужской",
            "female": "Женский",
            "unisex": "Унисекс",
        }
        normalized = (value or "").lower()
        return mapping.get(normalized, "Унисекс")

    async def _ensure_filters(user_id: int, state: FSMContext) -> FilterOptions:
        data = await state.get_data()
        return FilterOptions(
            gender=data.get("gender", "unisex"),
            age_bucket=data.get("age", "18-24"),
            style=data.get("style", "normal"),
        )

    async def _send_models(
        message: Message, user_id: int, filters: FilterOptions, state: FSMContext
    ) -> None:
        profile = await repository.ensure_user(user_id)
        seen_ids = set(profile.seen_models if profile else [])
        gender = _catalog_gender(filters.gender)
        try:
            models = await catalog.pick_four(gender, seen_ids)
        except CatalogError as exc:
            logger.error("%s Failed to fetch catalog: %s", EVENT_ID["MODELS_SENT"], exc)
            await message.answer(msg.CATALOG_TEMPORARILY_UNAVAILABLE)
            await state.update_data(current_models=[])
            return
        if not models:
            await message.answer(msg.CATALOG_TEMPORARILY_UNAVAILABLE)
            await state.update_data(current_models=[])
            return
        await repository.add_seen_models(user_id, [model.unique_id for model in models])
        await state.update_data(current_models=models)
        await _send_model_pairs(message, models)
        logger.info("%s Sent %s models", EVENT_ID["MODELS_SENT"], len(models))

    async def _send_model_pairs(message: Message, models: list[GlassModel]) -> None:
        pairs = _pair_models(models)
        total_pairs = len(pairs)
        for index, pair in enumerate(pairs, start=1):
            caption = f"Подборка {index}/{total_pairs}" if total_pairs > 1 else ""
            try:
                await _send_pair_message(message, pair, caption)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s Failed to send model pair %s: %s",
                    EVENT_ID["MODELS_SENT"],
                    [model.unique_id for model in pair],
                    exc,
                )

    async def _send_pair_message(
        message: Message, pair: tuple[GlassModel, ...], caption: str
    ) -> None:
        caption_to_use = caption or None
        collage_result: CollageResult | None = None
        if collage.enabled:
            collage_result = await collage.build_collage([model.img_user_url for model in pair])
        if collage_result and collage_result.included_indices:
            included_models = [pair[i] for i in collage_result.included_indices]
            keyboard = pair_selection_keyboard(
                [(model.unique_id, model.title) for model in included_models]
            )
            filename = f"collage-{uuid.uuid4().hex}.jpg"
            collage_caption = caption_to_use if 0 in collage_result.included_indices else None
            await message.answer_photo(
                photo=BufferedInputFile(collage_result.image_bytes, filename=filename),
                caption=collage_caption,
                reply_markup=keyboard,
            )
            missing_indices = set(range(len(pair))) - set(collage_result.included_indices)
            for missing_index in missing_indices:
                model = pair[missing_index]
                await _send_single_model(message, model, caption if missing_index == 0 else "")
            return

        for offset, model in enumerate(pair):
            await _send_single_model(message, model, caption if offset == 0 else "")

    async def _send_single_model(message: Message, model: GlassModel, caption: str) -> None:
        keyboard = pair_selection_keyboard([(model.unique_id, model.title)])
        await message.answer_photo(
            photo=URLInputFile(model.img_user_url),
            caption=caption or None,
            reply_markup=keyboard,
        )

    def _pair_models(models: list[GlassModel]) -> list[tuple[GlassModel, ...]]:
        return [tuple(models[i : i + 2]) for i in range(0, len(models), 2)]

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
        await message.bot.download(photo, destination=path)
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
        await message.answer("Ищу свежие модели...")
        await _send_models(message, user_id, filters, state)
        logger.info("%s Photo received from %s", EVENT_ID["PHOTO_RECEIVED"], user_id)

    @router.callback_query(StateFilter(TryOnStates.SHOW_RECS), F.data.startswith("pick|"))
    async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
        model_id = callback.data.replace("pick|", "")
        data = await state.get_data()
        models_data: List[GlassModel] = data.get("current_models", [])
        selected = next((model for model in models_data if model.unique_id == model_id), None)
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

    async def _perform_generation(message: Message, state: FSMContext, model: GlassModel) -> None:
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
                overlays=[],
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
        if not results:
            await message.answer(msg.GENERATION_FAILED, reply_markup=retry_keyboard())
            await state.set_state(TryOnStates.ERROR)
            return
        primary_result = results[0]
        remaining = await repository.remaining_tries(user_id)
        keyboard = generation_result_keyboard(model.site_url, remaining)
        await message.answer_photo(
            FSInputFile(path=str(primary_result)),
            caption=msg.GENERATION_SUCCESS,
            reply_markup=keyboard,
        )
        if remaining <= 0:
            await state.set_state(TryOnStates.LIMIT_REACHED)
            await message.answer(msg.LIMIT_REACHED, reply_markup=limit_reached_keyboard())
            logger.info("%s Limit reached post generation %s", EVENT_ID["LIMIT_REACHED"], user_id)
        logger.info("%s Generation succeeded for %s", EVENT_ID["GENERATION_SUCCESS"], user_id)

    @router.callback_query(StateFilter(TryOnStates.RESULT), F.data.startswith("more|"))
    async def result_more(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.LIMIT_REACHED)
            await callback.message.answer(msg.LIMIT_REACHED, reply_markup=limit_reached_keyboard())
            await callback.answer()
            return
        filters = await _ensure_filters(user_id, state)
        await state.set_state(TryOnStates.SHOW_RECS)
        await callback.message.answer("Ищу свежие модели...")
        await _send_models(callback.message, user_id, filters, state)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.LIMIT_REACHED), F.data == "limit_promo")
    async def limit_promo(callback: CallbackQuery) -> None:
        await callback.message.answer(msg.PROMO_CODE)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.LIMIT_REACHED), F.data == "limit_remind")
    async def limit_remind(callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        when = datetime.now(timezone.utc) + timedelta(hours=reminder_hours)
        await repository.set_reminder(user_id, when)
        await callback.message.answer("Напомню через сутки!")
        await callback.answer()
        logger.info(
            "%s Reminder scheduled for %s", EVENT_ID["REMINDER_SCHEDULED"], user_id
        )

    @router.callback_query(StateFilter(TryOnStates.ERROR), F.data == "retry")
    async def retry(callback: CallbackQuery, state: FSMContext) -> None:
        data = await state.get_data()
        selected: Optional[GlassModel] = data.get("selected_model")
        if not selected:
            await callback.answer("Нет выбранной модели", show_alert=True)
            return
        await state.set_state(TryOnStates.GENERATING)
        await callback.message.answer(msg.GENERATING)
        await callback.answer()
        await _perform_generation(callback.message, state, selected)

    return router
