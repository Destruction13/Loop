
"""FSM definitions and handler registration."""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Sequence

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
)
from aiogram.types.input_file import BufferedInputFile, FSInputFile, URLInputFile

from app.keyboards import (
    all_seen_keyboard,
    batch_selection_keyboard,
    gender_keyboard,
    generation_result_keyboard,
    limit_reached_keyboard,
    promo_keyboard,
    retry_keyboard,
    start_keyboard,
)
from app.logging_conf import EVENT_ID
from app.models import FilterOptions, GlassModel
from app.services.catalog_base import CatalogError
from app.config import CollageConfig
from app.services.collage import (
    CollageProcessingError,
    CollageSourceUnavailable,
    build_three_tile_collage,
)
from app.services.repository import Repository
from app.services.storage_base import StorageService
from app.services.tryon_base import TryOnService
from app.services.recommendation import RecommendationService
from app.texts import messages as msg


class TryOnStates(StatesGroup):
    START = State()
    FOR_WHO = State()
    AWAITING_PHOTO = State()
    SHOW_RECS = State()
    GENERATING = State()
    RESULT = State()
    DAILY_LIMIT_REACHED = State()
    ERROR = State()


class GenerationOutcome(Enum):
    """Post-generation follow-up decision."""

    FIRST = "first"
    FOLLOWUP = "followup"
    LIMIT = "limit"


@dataclass(slots=True)
class GenerationPlan:
    """Information about the follow-up flow after generation."""

    outcome: GenerationOutcome
    remaining: int


def resolve_generation_followup(
    *, first_generated_today: bool, remaining: int
) -> GenerationPlan:
    """Resolve which follow-up path to use after generation."""

    if remaining <= 0:
        return GenerationPlan(outcome=GenerationOutcome.LIMIT, remaining=0)
    outcome = GenerationOutcome.FIRST if first_generated_today else GenerationOutcome.FOLLOWUP
    return GenerationPlan(outcome=outcome, remaining=remaining)


def next_first_flag_value(current_flag: bool, outcome: GenerationOutcome) -> bool:
    """Return the next value for the first-generation flag."""

    if current_flag and outcome in {GenerationOutcome.FIRST, GenerationOutcome.LIMIT}:
        return False
    return current_flag


def chunk_models(
    models: Sequence[GlassModel], chunk_size: int
) -> list[tuple[GlassModel, ...]]:
    """Split models into consecutive chunks preserving order."""

    if chunk_size <= 0:
        return []
    return [tuple(models[i : i + chunk_size]) for i in range(0, len(models), chunk_size)]


def setup_router(
    *,
    repository: Repository,
    recommender: RecommendationService,
    tryon: TryOnService,
    storage: StorageService,
    collage_config: CollageConfig,
    collage_builder: Callable[[Sequence[str | None], CollageConfig], Awaitable[io.BytesIO]] = build_three_tile_collage,
    batch_size: int,
    reminder_hours: int,
    selection_button_title_max: int,
    landing_url: str,
    promo_code: str,
    no_more_message_key: str,
) -> Router:
    router = Router()
    logger = logging.getLogger("loop_bot.handlers")

    async def _ensure_filters(user_id: int, state: FSMContext) -> FilterOptions:
        data = await state.get_data()
        return FilterOptions(gender=data.get("gender", "unisex"))

    batch_source = "batch3"

    async def _send_models(
        message: Message, user_id: int, filters: FilterOptions, state: FSMContext
    ) -> None:
        try:
            result = await recommender.recommend_for_user(user_id, filters.gender)
        except CatalogError as exc:
            logger.error("%s Failed to fetch catalog: %s", EVENT_ID["MODELS_SENT"], exc)
            await message.answer(msg.CATALOG_TEMPORARILY_UNAVAILABLE)
            await state.update_data(current_models=[])
            await _delete_state_message(message, state, "preload_message_id")
            return
        if not result.models:
            try:
                marketing_message = msg.marketing_text(no_more_message_key)
            except KeyError:
                marketing_message = msg.CATALOG_TEMPORARILY_UNAVAILABLE
            await message.answer(
                marketing_message,
                reply_markup=all_seen_keyboard(landing_url),
            )
            await state.update_data(current_models=[], last_batch=[])
            await _delete_state_message(message, state, "preload_message_id")
            return
        batch = list(result.models)
        await state.update_data(current_models=batch, last_batch=batch)
        await _send_model_batches(message, batch)
        await _delete_state_message(message, state, "preload_message_id")
        if result.exhausted:
            try:
                marketing_message = msg.marketing_text(no_more_message_key)
            except KeyError:
                marketing_message = msg.CATALOG_TEMPORARILY_UNAVAILABLE
            await message.answer(
                marketing_message,
                reply_markup=all_seen_keyboard(landing_url),
            )

    async def _send_model_batches(message: Message, batch: list[GlassModel]) -> None:
        groups = chunk_models(batch, batch_size)
        total_groups = len(groups) if groups else 0
        for index, group in enumerate(groups, start=1):
            label_text = msg.BATCH_TITLE.format(index=index, total=total_groups)
            try:
                await _send_batch_message(message, group, label_text)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s Failed to send model batch %s: %s",
                    EVENT_ID["MODELS_SENT"],
                    [model.unique_id for model in group],
                    exc,
                )

    async def _send_batch_message(
        message: Message, group: tuple[GlassModel, ...], label_text: str
    ) -> None:
        keyboard = batch_selection_keyboard(
            [(item.unique_id, item.title) for item in group],
            source=batch_source,
            max_title_length=selection_button_title_max,
        )
        urls = [item.img_user_url for item in group]
        try:
            buffer = await collage_builder(urls, collage_config)
        except CollageSourceUnavailable:
            await message.answer(
                f"{label_text}\n\n{msg.COLLAGE_IMAGES_UNAVAILABLE}",
                reply_markup=keyboard,
            )
            return
        except CollageProcessingError as exc:
            logger.warning(
                "Collage processing failed for models %s: %s",
                [model.unique_id for model in group],
                exc,
            )
            await _send_batch_as_photos(
                message, group, label_text=label_text, reply_markup=keyboard
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected collage error for models %s: %s",
                [model.unique_id for model in group],
                exc,
            )
            await _send_batch_as_photos(
                message, group, label_text=label_text, reply_markup=keyboard
            )
            return

        filename = f"collage-{uuid.uuid4().hex}.jpg"
        collage_bytes = buffer.getvalue()
        buffer.close()
        await message.answer_photo(
            photo=BufferedInputFile(collage_bytes, filename=filename),
            caption=label_text,
            reply_markup=keyboard,
        )
        logger.info(
            "%s Collage %sx%s sent for models %s",
            EVENT_ID["MODELS_SENT"],
            collage_config.width,
            collage_config.height,
            [model.unique_id for model in group],
        )

    async def _send_batch_as_photos(
        message: Message,
        group: tuple[GlassModel, ...],
        *,
        label_text: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        last_index = len(group) - 1
        for index, item in enumerate(group):
            caption = label_text if index == last_index else None
            markup = reply_markup if index == last_index else None
            await message.answer_photo(
                photo=URLInputFile(item.img_user_url),
                caption=caption,
                reply_markup=markup,
            )

    async def _delete_state_message(message: Message, state: FSMContext, key: str) -> None:
        data = await state.get_data()
        message_id = data.get(key)
        if not message_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, message_id)
        except TelegramBadRequest as exc:
            logger.warning("Failed to delete %s message %s: %s", key, message_id, exc)
        finally:
            await state.update_data(**{key: None})

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
        await message.answer(msg.START_WELCOME, reply_markup=start_keyboard())
        logger.info("%s User %s entered start", EVENT_ID["START"], message.from_user.id)

    @router.callback_query(StateFilter(TryOnStates.START), F.data == "start_go")
    async def start_go(callback: CallbackQuery, state: FSMContext) -> None:
        await state.set_state(TryOnStates.FOR_WHO)
        await callback.message.edit_text(
            msg.START_GENDER_PROMPT, reply_markup=gender_keyboard()
        )
        await state.update_data(gender_prompt_message_id=callback.message.message_id)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.START), F.data == "start_info")
    async def start_info(callback: CallbackQuery) -> None:
        await callback.answer(msg.START_MAGIC_INFO, show_alert=True)

    @router.callback_query(StateFilter(TryOnStates.FOR_WHO))
    async def select_gender(callback: CallbackQuery, state: FSMContext) -> None:
        gender = callback.data.replace("gender_", "")
        await repository.update_filters(callback.from_user.id, gender=gender)
        await state.update_data(gender=gender, first_generated_today=True)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await _delete_state_message(callback.message, state, "gender_prompt_message_id")
        await callback.message.answer(
            msg.PHOTO_INSTRUCTION,
            reply_markup=ReplyKeyboardRemove(),
        )
        await callback.answer()
        logger.info("%s Gender selected %s", EVENT_ID["FILTER_SELECTED"], gender)

    @router.message(StateFilter(TryOnStates.AWAITING_PHOTO, TryOnStates.RESULT), ~F.photo)
    async def reject_non_photo(message: Message) -> None:
        await message.answer(msg.NOT_PHOTO_WARNING)

    @router.message(StateFilter(TryOnStates.AWAITING_PHOTO, TryOnStates.RESULT), F.photo)
    async def accept_photo(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id
        photo = message.photo[-1]
        filename = f"{photo.file_unique_id}.jpg"
        path = await storage.allocate_upload_path(user_id, filename)
        await message.bot.download(photo, destination=path)
        await state.update_data(upload=str(path))
        profile = await repository.ensure_daily_reset(user_id)
        if profile.daily_used == 0:
            await state.update_data(first_generated_today=True)
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await message.answer(
                msg.DAILY_LIMIT_MESSAGE,
                reply_markup=limit_reached_keyboard(landing_url),
            )
            logger.info("%s Limit reached for user %s", EVENT_ID["LIMIT_REACHED"], user_id)
            return
        filters = await _ensure_filters(user_id, state)
        await state.set_state(TryOnStates.SHOW_RECS)
        preload_message = await message.answer(
            msg.SEARCHING_MODELS_PROMPT,
            reply_markup=ReplyKeyboardRemove(),
        )
        await state.update_data(preload_message_id=preload_message.message_id)
        await _send_models(message, user_id, filters, state)
        logger.info("%s Photo received from %s", EVENT_ID["PHOTO_RECEIVED"], user_id)

    @router.callback_query(StateFilter(TryOnStates.SHOW_RECS), F.data.startswith("pick:"))
    async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
        parts = callback.data.split(":", 2)
        if len(parts) == 3:
            _, batch_source_key, model_id = parts
        else:  # fallback for legacy format
            batch_source_key = "unknown"
            model_id = callback.data.replace("pick:", "", 1)
        data = await state.get_data()
        models_data: List[GlassModel] = data.get("current_models", [])
        selected = next((model for model in models_data if model.unique_id == model_id), None)
        if not selected:
            await callback.answer(msg.MODEL_UNAVAILABLE_ALERT, show_alert=True)
            return
        logger.debug(
            "User %s selected model %s from %s",
            callback.from_user.id,
            model_id,
            batch_source_key,
        )
        remaining = await repository.remaining_tries(callback.from_user.id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await callback.message.answer(
                msg.DAILY_LIMIT_MESSAGE,
                reply_markup=limit_reached_keyboard(landing_url),
            )
            await callback.answer()
            return
        await state.update_data(selected_model=selected, last_batch=[])
        await state.set_state(TryOnStates.GENERATING)
        generation_message = await callback.message.answer(msg.GENERATING_PROMPT)
        await state.update_data(generation_message_id=generation_message.message_id)
        await callback.answer()
        await _perform_generation(callback.message, state, selected)

    async def _perform_generation(message: Message, state: FSMContext, model: GlassModel) -> None:
        user_id = message.chat.id
        data = await state.get_data()
        upload_value = data.get("upload")
        if not upload_value:
            await _delete_state_message(message, state, "generation_message_id")
            await message.answer(msg.GENERATION_FAILED, reply_markup=retry_keyboard())
            await state.set_state(TryOnStates.ERROR)
            return
        upload_path = Path(upload_value)
        try:
            results = await tryon.generate(
                user_id=user_id,
                session_id=uuid.uuid4().hex,
                input_photo=upload_path,
                overlays=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s Generation failed: %s", EVENT_ID["GENERATION_FAILED"], exc
            )
            await _delete_state_message(message, state, "generation_message_id")
            await message.answer(msg.GENERATION_FAILED, reply_markup=retry_keyboard())
            await state.set_state(TryOnStates.ERROR)
            return
        if not results:
            await _delete_state_message(message, state, "generation_message_id")
            await message.answer(msg.GENERATION_FAILED, reply_markup=retry_keyboard())
            await state.set_state(TryOnStates.ERROR)
            return
        primary_result = results[0]
        await repository.inc_used_on_success(user_id)
        remaining = await repository.remaining_tries(user_id)
        plan = resolve_generation_followup(
            first_generated_today=data.get("first_generated_today", True),
            remaining=remaining,
        )
        caption_source = (
            msg.FIRST_RESULT_CAPTION
            if data.get("first_generated_today", True)
            else msg.NEXT_RESULT_CAPTION
        )
        if isinstance(caption_source, (list, tuple)):
            caption_text = "".join(caption_source)
        else:
            caption_text = str(caption_source)
        await message.answer_photo(
            FSInputFile(path=str(primary_result)),
            caption=caption_text,
            reply_markup=generation_result_keyboard(model.site_url, plan.remaining),
        )
        await _delete_state_message(message, state, "generation_message_id")
        new_flag = next_first_flag_value(
            data.get("first_generated_today", True), plan.outcome
        )
        await state.update_data(first_generated_today=new_flag)
        if plan.outcome is GenerationOutcome.LIMIT:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await message.answer(
                msg.DAILY_LIMIT_MESSAGE,
                reply_markup=limit_reached_keyboard(landing_url),
            )
            logger.info(
                "%s Limit reached post generation %s", EVENT_ID["LIMIT_REACHED"], user_id
            )
        else:
            await state.set_state(TryOnStates.RESULT)
        logger.info("%s Generation succeeded for %s", EVENT_ID["GENERATION_SUCCESS"], user_id)

    @router.callback_query(StateFilter(TryOnStates.RESULT), F.data.startswith("more|"))
    async def result_more(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await callback.message.answer(
                msg.DAILY_LIMIT_MESSAGE,
                reply_markup=limit_reached_keyboard(landing_url),
            )
            await callback.answer()
            return
        filters = await _ensure_filters(user_id, state)
        await state.set_state(TryOnStates.SHOW_RECS)
        preload_message = await callback.message.answer(msg.SEARCHING_MODELS_PROMPT)
        await state.update_data(preload_message_id=preload_message.message_id)
        await _send_models(callback.message, user_id, filters, state)
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.DAILY_LIMIT_REACHED), F.data == "limit_promo")
    async def limit_promo(callback: CallbackQuery) -> None:
        text = msg.PROMO_MESSAGE_TEMPLATE.format(promo_code=promo_code)
        await callback.message.answer(
            text,
            reply_markup=promo_keyboard(landing_url),
        )
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.DAILY_LIMIT_REACHED), F.data == "limit_remind")
    async def limit_remind(callback: CallbackQuery) -> None:
        user_id = callback.from_user.id
        when = datetime.now(timezone.utc) + timedelta(hours=reminder_hours)
        await repository.set_reminder(user_id, when)
        await callback.message.answer(msg.REMINDER_CONFIRMATION)
        await callback.answer()
        logger.info(
            "%s Reminder scheduled for %s", EVENT_ID["REMINDER_SCHEDULED"], user_id
        )

    @router.callback_query(F.data == "reminder_go")
    async def reminder_go(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        profile = await repository.ensure_user(user_id)
        await repository.set_reminder(user_id, None)
        if not profile.gender:
            await state.set_state(TryOnStates.START)
            await callback.message.answer(msg.START_WELCOME, reply_markup=start_keyboard())
            await callback.answer()
            return
        first_flag = profile.daily_used == 0
        await state.update_data(gender=profile.gender, first_generated_today=first_flag)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await callback.message.answer(
            msg.PHOTO_INSTRUCTION,
            reply_markup=ReplyKeyboardRemove(),
        )
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.ERROR), F.data == "retry")
    async def retry(callback: CallbackQuery, state: FSMContext) -> None:
        data = await state.get_data()
        selected: Optional[GlassModel] = data.get("selected_model")
        if not selected:
            await callback.answer("Нет выбранной модели", show_alert=True)
            return
        await state.set_state(TryOnStates.GENERATING)
        generation_message = await callback.message.answer(msg.GENERATING_PROMPT)
        await state.update_data(generation_message_id=generation_message.message_id)
        await callback.answer()
        await _perform_generation(callback.message, state, selected)

    return router
