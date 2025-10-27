
"""FSM definitions and handler registration."""

from __future__ import annotations

import io
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Sequence

from aiogram import BaseMiddleware, F, Router
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
    contact_request_keyboard,
    limit_reached_keyboard,
    promo_keyboard,
    retry_keyboard,
    start_keyboard,
)
from app.logging_conf import EVENT_ID
from app.models import FilterOptions, GlassModel, UserContact
from app.services.catalog_base import CatalogError
from app.config import CollageConfig
from app.services.collage import (
    CollageProcessingError,
    CollageSourceUnavailable,
    build_three_tile_collage,
)
from app.services.leads_export import LeadPayload, LeadsExporter
from app.services.repository import Repository
from app.services.storage_base import StorageService
from app.services.tryon_base import TryOnService
from app.services.recommendation import RecommendationService
from app.utils.phone import normalize_phone
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


class ContactRequest(StatesGroup):
    waiting_for_phone = State()


CONTACT_INITIAL_TRIGGER = 2
CONTACT_REMINDER_TRIGGER = 5


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
    site_url: str,
    promo_code: str,
    no_more_message_key: str,
    contact_reward_rub: int,
    promo_contact_code: str,
    leads_exporter: LeadsExporter,
) -> Router:
    router = Router()
    logger = logging.getLogger("loop_bot.handlers")

    class ActivityMiddleware(BaseMiddleware):
        """Tracks user activity timestamps."""

        def __init__(self, repository: Repository) -> None:
            super().__init__()
            self._repository = repository

        async def __call__(self, handler, event, data):
            user = data.get("event_from_user")
            if user:
                await self._repository.touch_activity(user.id)
            return await handler(event, data)

    activity_middleware = ActivityMiddleware(repository)
    router.message.middleware(activity_middleware)
    router.callback_query.middleware(activity_middleware)

    async def _ensure_filters(user_id: int, state: FSMContext) -> FilterOptions:
        data = await state.get_data()
        return FilterOptions(gender=data.get("gender", "unisex"))

    batch_source = "src=batch2"

    async def _delete_last_aux_message(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        message_id = data.get("last_aux_message_id")
        if not message_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, message_id)
        except TelegramBadRequest as exc:
            logger.debug(
                "Failed to delete last auxiliary message %s: %s", message_id, exc
            )
        finally:
            await state.update_data(last_aux_message_id=None)

    async def _send_aux_message(
        source_message: Message,
        state: FSMContext,
        send_method: Callable[..., Awaitable[Message]],
        *args,
        **kwargs,
    ) -> Message:
        await _delete_last_aux_message(source_message, state)
        sent_message = await send_method(*args, **kwargs)
        await state.update_data(last_aux_message_id=sent_message.message_id)
        return sent_message

    async def _send_delivery_message(
        source_message: Message,
        state: FSMContext,
        send_method: Callable[..., Awaitable[Message]],
        *args,
        **kwargs,
    ) -> Message:
        return await send_method(*args, **kwargs)

    def _remove_more_button_from_markup(
        markup: InlineKeyboardMarkup | None,
    ) -> InlineKeyboardMarkup | None:
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

    async def _maybe_request_contact(
        message: Message,
        state: FSMContext,
        user_id: int,
        *,
        origin_state: Optional[str] = None,
    ) -> bool:
        data = await state.get_data()
        if data.get("contact_request_active"):
            return True
        profile = await repository.ensure_user(user_id)
        if profile.contact_never:
            return False
        contact = await repository.get_user_contact(user_id)
        if contact and contact.consent:
            return False
        if profile.gen_count < CONTACT_INITIAL_TRIGGER:
            return False
        if profile.contact_skip_once:
            if profile.gen_count >= CONTACT_REMINDER_TRIGGER:
                await repository.set_contact_skip_once(user_id, False)
            else:
                return False
        current_state = origin_state or await state.get_state()
        pending_state = data.get("contact_pending_result_state")
        if not pending_state and current_state == TryOnStates.RESULT.state:
            pending_state = "result"
        prompt_text = (
            f"<b>{msg.ASK_PHONE_TITLE}</b>\n\n"
            f"{msg.ASK_PHONE_BODY.format(rub=contact_reward_rub)}\n\n"
            f"{msg.ASK_PHONE_PROMPT_MANUAL}"
        )
        update_payload = {
            "contact_request_active": True,
            "contact_pending_generation": True,
        }
        if pending_state and pending_state != data.get("contact_pending_result_state"):
            update_payload["contact_pending_result_state"] = pending_state
        await state.update_data(**update_payload)
        await state.set_state(ContactRequest.waiting_for_phone)
        await _send_aux_message(
            message,
            state,
            message.answer,
            prompt_text,
            reply_markup=contact_request_keyboard(),
        )
        logger.info("%s Contact requested for %s", EVENT_ID["MODELS_SENT"], user_id)
        return True

    async def _send_models(
        message: Message,
        user_id: int,
        filters: FilterOptions,
        state: FSMContext,
        *,
        skip_contact_prompt: bool = False,
    ) -> bool:
        if not skip_contact_prompt:
            if await _maybe_request_contact(message, state, user_id):
                return False
        try:
            result = await recommender.recommend_for_user(user_id, filters.gender)
        except CatalogError as exc:
            logger.error("%s Failed to fetch catalog: %s", EVENT_ID["MODELS_SENT"], exc)
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.CATALOG_TEMPORARILY_UNAVAILABLE,
            )
            await state.update_data(current_models=[])
            await _delete_state_message(message, state, "preload_message_id")
            return False
        if not result.models:
            try:
                marketing_message = msg.marketing_text(no_more_message_key)
            except KeyError:
                marketing_message = msg.CATALOG_TEMPORARILY_UNAVAILABLE
            await _send_aux_message(
                message,
                state,
                message.answer,
                marketing_message,
                reply_markup=all_seen_keyboard(site_url),
            )
            await state.update_data(current_models=[], last_batch=[])
            await _delete_state_message(message, state, "preload_message_id")
            return False
        batch = list(result.models)
        await state.update_data(current_models=batch, last_batch=batch)
        await _send_model_batches(message, state, batch)
        await _delete_state_message(message, state, "preload_message_id")
        if result.exhausted:
            try:
                marketing_message = msg.marketing_text(no_more_message_key)
            except KeyError:
                marketing_message = msg.CATALOG_TEMPORARILY_UNAVAILABLE
            await _send_aux_message(
                message,
                state,
                message.answer,
                marketing_message,
                reply_markup=all_seen_keyboard(site_url),
            )
        return True

    async def _send_model_batches(
        message: Message, state: FSMContext, batch: list[GlassModel]
    ) -> None:
        groups = chunk_models(batch, batch_size)
        for group in groups:
            try:
                await _send_batch_message(message, state, group)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s Failed to send model batch %s: %s",
                    EVENT_ID["MODELS_SENT"],
                    [model.unique_id for model in group],
                    exc,
                )

    async def _resume_after_contact(
        message: Message,
        state: FSMContext,
        *,
        send_generation: bool,
    ) -> None:
        data = await state.get_data()
        user_id = message.from_user.id
        pending_state = data.get("contact_pending_result_state")
        await state.update_data(
            contact_request_active=False,
            contact_pending_result_state=None,
        )
        if pending_state == "limit":
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
        elif pending_state == "result":
            await state.set_state(TryOnStates.RESULT)
        else:
            await state.set_state(TryOnStates.SHOW_RECS)
        pending_generation = data.get("contact_pending_generation", False)
        allow_generation = (
            send_generation and pending_generation and pending_state != "limit"
        )
        if allow_generation:
            await state.update_data(contact_pending_generation=False)
            filters = await _ensure_filters(user_id, state)
            await _send_models(
                message,
                user_id,
                filters,
                state,
                skip_contact_prompt=True,
            )
        else:
            await state.update_data(contact_pending_generation=False)

    async def _export_lead(
        user_id: int,
        phone_e164: str,
        source: str,
        consent_ts: int,
        *,
        username: str | None,
        full_name: str | None,
    ) -> None:
        payload = LeadPayload(
            tg_user_id=user_id,
            phone_e164=phone_e164,
            source=source,
            consent_ts=consent_ts,
            username=username,
            full_name=full_name,
        )
        await leads_exporter.export_lead_to_sheet(payload)

    async def _store_contact(
        message: Message,
        state: FSMContext,
        phone_e164: str,
        *,
        source: str,
    ) -> None:
        user = message.from_user
        user_id = user.id
        existing = await repository.get_user_contact(user_id)
        consent_ts = int(time.time())
        contact = UserContact(
            tg_user_id=user_id,
            phone_e164=phone_e164,
            source=source,
            consent=True,
            consent_ts=consent_ts,
            reward_granted=existing.reward_granted if existing else False,
        )
        changed = existing is None or existing.phone_e164 != phone_e164
        reward_needed = existing is None or not existing.reward_granted or changed
        if reward_needed:
            contact.reward_granted = True
        await repository.upsert_user_contact(contact)
        await repository.set_contact_skip_once(user_id, False)
        await repository.set_contact_never(user_id, False)
        full_name = getattr(user, "full_name", None)
        username = getattr(user, "username", None)
        if changed:
            await _export_lead(
                user_id,
                phone_e164,
                source,
                consent_ts,
                username=username,
                full_name=full_name,
            )
        if reward_needed:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_THANKS.format(
                    rub=contact_reward_rub, promo=promo_contact_code
                ),
                reply_markup=ReplyKeyboardRemove(),
            )
            logger.info(
                "%s Contact stored for %s via %s",
                EVENT_ID["MODELS_SENT"],
                user_id,
                source,
            )
            await _resume_after_contact(message, state, send_generation=False)
            current_state = await state.get_state()
            if current_state != TryOnStates.DAILY_LIMIT_REACHED.state:
                caption_source = msg.NEXT_RESULT_CAPTION
                if isinstance(caption_source, (list, tuple)):
                    followup_text = "".join(caption_source)
                else:
                    followup_text = str(caption_source)
                await state.set_state(TryOnStates.AWAITING_PHOTO)
                await state.update_data(
                    upload=None,
                    current_models=[],
                    last_batch=[],
                    preload_message_id=None,
                    generation_message_id=None,
                )
                await _send_aux_message(
                    message,
                    state,
                    message.answer,
                    followup_text,
                )
            return
        else:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_ALREADY_HAVE,
                reply_markup=ReplyKeyboardRemove(),
            )
            logger.info(
                "%s Contact already existed for %s",
                EVENT_ID["MODELS_SENT"],
                user_id,
            )
        await _resume_after_contact(message, state, send_generation=True)

    async def _handle_manual_phone(
        message: Message, state: FSMContext, *, source: str
    ) -> None:
        raw = (message.text or "").strip()
        normalized = normalize_phone(raw)
        if not normalized:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_INVALID,
            )
            return
        await _store_contact(message, state, normalized, source=source)

    async def _send_batch_message(
        message: Message, state: FSMContext, group: tuple[GlassModel, ...]
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
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.COLLAGE_IMAGES_UNAVAILABLE,
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
                message,
                state,
                group,
                reply_markup=keyboard,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected collage error for models %s: %s",
                [model.unique_id for model in group],
                exc,
            )
            await _send_batch_as_photos(
                message,
                state,
                group,
                reply_markup=keyboard,
            )
            return

        filename = f"collage-{uuid.uuid4().hex}.jpg"
        collage_bytes = buffer.getvalue()
        buffer.close()
        await _send_delivery_message(
            message,
            state,
            message.answer_photo,
            photo=BufferedInputFile(collage_bytes, filename=filename),
            caption=None,
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
        state: FSMContext,
        group: tuple[GlassModel, ...],
        *,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        last_index = len(group) - 1
        for index, item in enumerate(group):
            caption = None
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
            if data.get("last_aux_message_id") == message_id:
                await state.update_data(last_aux_message_id=None)
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
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.START_WELCOME,
            reply_markup=start_keyboard(),
        )
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
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.PHOTO_INSTRUCTION,
            reply_markup=ReplyKeyboardRemove(),
        )
        await callback.answer()
        logger.info("%s Gender selected %s", EVENT_ID["FILTER_SELECTED"], gender)

    @router.message(StateFilter(TryOnStates.AWAITING_PHOTO, TryOnStates.RESULT), ~F.photo)
    async def reject_non_photo(message: Message, state: FSMContext) -> None:
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.NOT_PHOTO_WARNING,
        )

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
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.DAILY_LIMIT_MESSAGE,
                reply_markup=limit_reached_keyboard(site_url),
            )
            logger.info("%s Limit reached for user %s", EVENT_ID["LIMIT_REACHED"], user_id)
            return
        filters = await _ensure_filters(user_id, state)
        await state.set_state(TryOnStates.SHOW_RECS)
        if await _maybe_request_contact(message, state, user_id):
            logger.info("%s Contact request queued for %s", EVENT_ID["MODELS_SENT"], user_id)
            return
        preload_message = await _send_aux_message(
            message,
            state,
            message.answer,
            msg.SEARCHING_MODELS_PROMPT,
            reply_markup=ReplyKeyboardRemove(),
        )
        await state.update_data(preload_message_id=preload_message.message_id)
        await _send_models(
            message,
            user_id,
            filters,
            state,
            skip_contact_prompt=True,
        )
        logger.info("%s Photo received from %s", EVENT_ID["PHOTO_RECEIVED"], user_id)

    @router.callback_query(
        StateFilter(TryOnStates.SHOW_RECS, TryOnStates.RESULT),
        F.data.startswith("pick:"),
    )
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
            await _send_aux_message(
                callback.message,
                state,
                callback.message.answer,
                msg.DAILY_LIMIT_MESSAGE,
            reply_markup=limit_reached_keyboard(site_url),
            )
            await callback.answer()
            return
        await callback.answer()
        try:
            await callback.message.delete()
        except TelegramBadRequest as exc:
            logger.warning(
                "Failed to delete recommendation message %s: %s",
                callback.message.message_id,
                exc,
            )
        await state.update_data(selected_model=selected)
        await state.set_state(TryOnStates.GENERATING)
        generation_message = await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.GENERATING_PROMPT,
        )
        await state.update_data(generation_message_id=generation_message.message_id)
        await _perform_generation(callback.message, state, selected)

    @router.message(StateFilter(ContactRequest.waiting_for_phone), F.contact)
    async def contact_shared(message: Message, state: FSMContext) -> None:
        contact = message.contact
        if not contact or not contact.phone_number:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_INVALID,
            )
            return
        normalized = normalize_phone(contact.phone_number)
        if not normalized:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_INVALID,
            )
            return
        await _store_contact(message, state, normalized, source="share_button")

    @router.message(StateFilter(ContactRequest.waiting_for_phone), F.text)
    async def contact_text(message: Message, state: FSMContext) -> None:
        text = (message.text or "").strip()
        user_id = message.from_user.id
        if text == msg.ASK_PHONE_BUTTON_SKIP:
            await repository.set_contact_skip_once(user_id, True)
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_SKIP_ACK,
                reply_markup=ReplyKeyboardRemove(),
            )
            await _resume_after_contact(message, state, send_generation=True)
            logger.info("%s Contact skip once for %s", EVENT_ID["MODELS_SENT"], user_id)
            return
        if text == msg.ASK_PHONE_BUTTON_NEVER:
            await repository.set_contact_never(user_id, True)
            await repository.set_contact_skip_once(user_id, False)
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_NEVER_ACK,
                reply_markup=ReplyKeyboardRemove(),
            )
            await _resume_after_contact(message, state, send_generation=True)
            logger.info("%s Contact opt-out for %s", EVENT_ID["MODELS_SENT"], user_id)
            return
        await _handle_manual_phone(message, state, source="manual")

    @router.message(StateFilter(ContactRequest.waiting_for_phone))
    async def contact_fallback(message: Message, state: FSMContext) -> None:
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.ASK_PHONE_INVALID,
        )

    async def _perform_generation(message: Message, state: FSMContext, model: GlassModel) -> None:
        user_id = message.chat.id
        data = await state.get_data()
        upload_value = data.get("upload")
        if not upload_value:
            await _delete_state_message(message, state, "generation_message_id")
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.GENERATION_FAILED,
                reply_markup=retry_keyboard(),
            )
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
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.GENERATION_FAILED,
                reply_markup=retry_keyboard(),
            )
            await state.set_state(TryOnStates.ERROR)
            return
        if not results:
            await _delete_state_message(message, state, "generation_message_id")
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.GENERATION_FAILED,
                reply_markup=retry_keyboard(),
            )
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
        result_markup = generation_result_keyboard(
            model.site_url, plan.remaining
        )
        if plan.outcome is GenerationOutcome.LIMIT:
            caption_text = f"{caption_text}\n\n{msg.DAILY_LIMIT_MESSAGE}"
            result_markup = limit_reached_keyboard(site_url)
        await _send_delivery_message(
            message,
            state,
            message.answer_photo,
            FSInputFile(path=str(primary_result)),
            caption=caption_text,
            reply_markup=result_markup,
        )
        await repository.increment_generation_count(user_id)
        await _delete_state_message(message, state, "generation_message_id")
        new_flag = next_first_flag_value(
            data.get("first_generated_today", True), plan.outcome
        )
        await state.update_data(first_generated_today=new_flag)
        contact_active = data.get("contact_request_active", False)
        if plan.outcome is GenerationOutcome.LIMIT:
            if contact_active:
                await state.update_data(contact_pending_result_state="limit")
            else:
                await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            logger.info(
                "%s Limit reached post generation %s", EVENT_ID["LIMIT_REACHED"], user_id
            )
        else:
            if contact_active:
                await state.update_data(contact_pending_result_state="result")
            else:
                await state.set_state(TryOnStates.RESULT)
        logger.info("%s Generation succeeded for %s", EVENT_ID["GENERATION_SUCCESS"], user_id)

    @router.callback_query(F.data.startswith("more|"))
    async def result_more(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        message = callback.message
        remove_source_message = callback.data in {"more|idle", "more|social"}
        if message:
            current_markup = getattr(message, "reply_markup", None)
            updated_markup = _remove_more_button_from_markup(current_markup)
            if updated_markup is not None:
                try:
                    await message.edit_reply_markup(reply_markup=updated_markup)
                except TelegramBadRequest as exc:
                    logger.debug(
                        "Failed to update keyboard for message %s: %s",
                        message.message_id,
                        exc,
                    )
        gen_count = await repository.get_generation_count(user_id)
        if gen_count >= 1:
            if remove_source_message and message:
                try:
                    await message.bot.delete_message(message.chat.id, message.message_id)
                except TelegramBadRequest as exc:
                    logger.debug(
                        "Failed to delete reminder message %s: %s",
                        message.message_id,
                        exc,
                    )
            prompt_source = msg.NEXT_RESULT_CAPTION
            if isinstance(prompt_source, (list, tuple)):
                prompt_text = "".join(prompt_source)
            else:
                prompt_text = str(prompt_source)
            await state.set_state(TryOnStates.AWAITING_PHOTO)
            await state.update_data(
                upload=None,
                current_models=[],
                last_batch=[],
                preload_message_id=None,
                generation_message_id=None,
            )
            if message:
                await _send_aux_message(
                    message,
                    state,
                    message.answer,
                    prompt_text,
                    reply_markup=ReplyKeyboardRemove(),
                )
            await callback.answer()
            return
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await _send_aux_message(
                callback.message,
                state,
                callback.message.answer,
                msg.DAILY_LIMIT_MESSAGE,
                reply_markup=limit_reached_keyboard(site_url),
            )
            await callback.answer()
            return
        filters = await _ensure_filters(user_id, state)
        previous_state = await state.get_state()
        if await _maybe_request_contact(
            callback.message, state, user_id, origin_state=previous_state
        ):
            await callback.answer()
            return
        await state.set_state(TryOnStates.SHOW_RECS)
        preload_message = await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.SEARCHING_MODELS_PROMPT,
        )
        await state.update_data(preload_message_id=preload_message.message_id)
        if remove_source_message and message:
            try:
                await message.bot.delete_message(message.chat.id, message.message_id)
            except TelegramBadRequest as exc:
                logger.debug(
                    "Failed to delete reminder message %s: %s",
                    message.message_id,
                    exc,
                )
        await _send_models(
            callback.message,
            user_id,
            filters,
            state,
            skip_contact_prompt=True,
        )
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.DAILY_LIMIT_REACHED), F.data == "limit_promo")
    async def limit_promo(callback: CallbackQuery, state: FSMContext) -> None:
        text = msg.PROMO_MESSAGE_TEMPLATE.format(promo_code=promo_code)
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            text,
            reply_markup=promo_keyboard(site_url),
        )
        await callback.answer()

    @router.callback_query(StateFilter(TryOnStates.DAILY_LIMIT_REACHED), F.data == "limit_remind")
    async def limit_remind(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        when = datetime.now(timezone.utc) + timedelta(hours=reminder_hours)
        await repository.set_reminder(user_id, when)
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.REMINDER_CONFIRMATION,
        )
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
            await _send_aux_message(
                callback.message,
                state,
                callback.message.answer,
                msg.START_WELCOME,
                reply_markup=start_keyboard(),
            )
            await callback.answer()
            return
        first_flag = profile.daily_used == 0
        await state.update_data(gender=profile.gender, first_generated_today=first_flag)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
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
        generation_message = await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.GENERATING_PROMPT,
        )
        await state.update_data(generation_message_id=generation_message.message_id)
        await callback.answer()
        await _perform_generation(callback.message, state, selected)

    return router
