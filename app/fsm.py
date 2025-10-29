
"""FSM definitions and handler registration."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from aiogram import BaseMiddleware, F, Router, Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.types.input_file import BufferedInputFile, FSInputFile, URLInputFile

from app.keyboards import (
    all_seen_keyboard,
    batch_selection_keyboard,
    gender_keyboard,
    generation_result_keyboard,
    contact_request_keyboard,
    limit_reached_keyboard,
    main_reply_keyboard,
    privacy_policy_keyboard,
    promo_keyboard,
    send_new_photo_keyboard,
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
from app.services.contact_export import ContactRecord, ContactSheetExporter
from app.services.leads_export import LeadPayload, LeadsExporter
from app.services.repository import Repository
from app.infrastructure.concurrency import with_generation_slot
from app.services.drive_fetch import fetch_drive_file
from app.services.image_io import (
    redownload_user_photo,
    resize_inplace,
    save_user_photo,
)
from app.services.nanobanana import (
    NanoBananaGenerationError,
    generate_glasses,
)
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
    contact_exporter: ContactSheetExporter,
    idle_nudge_seconds: int,
    enable_idle_nudge: bool,
    privacy_policy_url: str,
) -> Router:
    router = Router()
    logger = logging.getLogger("loop_bot.handlers")

    idle_delay = max(int(idle_nudge_seconds), 0)
    idle_enabled = enable_idle_nudge and idle_delay > 0
    idle_tasks: dict[int, asyncio.Task] = {}

    policy_url = (privacy_policy_url or "").strip()

    def _cancel_idle_timer(user_id: int) -> None:
        task = idle_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()

    def _should_schedule_idle(profile: UserProfile | None, now: datetime) -> bool:
        if not idle_enabled or profile is None:
            return False
        if not profile.gender:
            return False
        if profile.locked_until and profile.locked_until > now:
            return False
        if profile.nudge_sent_cycle:
            return False
        limit = profile.daily_try_limit if profile.daily_try_limit > 0 else None
        remaining = profile.remaining(limit)
        return remaining > 0

    def _extract_chat_id(event: Any) -> int | None:
        if isinstance(event, Message):
            return event.chat.id
        if isinstance(event, CallbackQuery) and event.message:
            return event.message.chat.id
        return None

    async def _delete_idle_nudge_message(
        state: FSMContext, bot: Bot, chat_id: int
    ) -> None:
        data = await state.get_data()
        message_id = data.get("idle_nudge_message_id")
        if not message_id:
            return
        try:
            await bot.delete_message(chat_id, message_id)
        except TelegramBadRequest as exc:
            logger.debug(
                "Failed to delete idle nudge message %s: %s", message_id, exc
            )
        finally:
            await state.update_data(idle_nudge_message_id=None)

    async def _idle_timeout_worker(
        user_id: int, chat_id: int, bot: Bot, state: FSMContext
    ) -> None:
        try:
            await asyncio.sleep(idle_delay)
            profile = await repository.ensure_daily_reset(user_id)
            now = datetime.now(timezone.utc)
            if not _should_schedule_idle(profile, now):
                return
            text = f"<b>{msg.IDLE_REMINDER_TITLE}</b>\n{msg.IDLE_REMINDER_BODY}"
            keyboard = idle_reminder_keyboard(site_url)
            try:
                message = await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=keyboard,
                )
            except (TelegramBadRequest, TelegramForbiddenError) as exc:
                logger.debug(
                    "Failed to deliver idle nudge to %s: %s", user_id, exc
                )
                return
            await repository.mark_cycle_nudge_sent(user_id)
            await state.update_data(idle_nudge_message_id=message.message_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Idle nudge task failed for %s", user_id)
        finally:
            idle_tasks.pop(user_id, None)

    async def _handle_idle_timer(
        user_id: int,
        event: Any,
        data: dict[str, Any],
        profile: UserProfile | None,
    ) -> None:
        _cancel_idle_timer(user_id)
        if not idle_enabled:
            return
        if profile is None:
            return
        now = datetime.now(timezone.utc)
        if not _should_schedule_idle(profile, now):
            return
        chat_id = _extract_chat_id(event)
        if chat_id is None:
            return
        bot: Bot | None = data.get("bot")
        state: FSMContext | None = data.get("state")
        if not bot or state is None:
            return
        idle_tasks[user_id] = asyncio.create_task(
            _idle_timeout_worker(user_id, chat_id, bot, state)
        )

    class ActivityMiddleware(BaseMiddleware):
        """Tracks user activity timestamps."""

        def __init__(self, repository: Repository) -> None:
            super().__init__()
            self._repository = repository

        async def __call__(self, handler, event, data):
            user = data.get("event_from_user")
            if not user:
                return await handler(event, data)
            user_id = user.id
            _cancel_idle_timer(user_id)
            await self._repository.ensure_daily_reset(user_id)
            await self._repository.touch_activity(user_id)
            try:
                return await handler(event, data)
            finally:
                if idle_enabled:
                    profile_after = await self._repository.ensure_daily_reset(user_id)
                    await _handle_idle_timer(user_id, event, data, profile_after)

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
        track: bool = True,
        delete_previous: bool = True,
        **kwargs,
    ) -> Message:
        if delete_previous:
            await _delete_last_aux_message(source_message, state)
        send_args: tuple[Any, ...] = args
        if send_args:
            first_arg = send_args[0]
            if isinstance(first_arg, (list, tuple)):
                send_args = ("".join(first_arg),) + send_args[1:]
        if "reply_markup" not in kwargs or kwargs["reply_markup"] is None:
            kwargs["reply_markup"] = main_reply_keyboard()
        sent_message = await send_method(*send_args, **kwargs)
        if track:
            await state.update_data(last_aux_message_id=sent_message.message_id)
        elif delete_previous:
            await state.update_data(last_aux_message_id=None)
        return sent_message

    async def _send_privacy_policy(
        message: Message, state: FSMContext
    ) -> None:
        markup = privacy_policy_keyboard(policy_url)
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.PRIVACY_POLICY_TEXT,
            reply_markup=markup,
        )

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

    def _render_text(source: str | Sequence[str]) -> str:
        if isinstance(source, (list, tuple)):
            return "".join(source)
        return str(source)

    async def _prompt_for_next_photo(
        message: Message,
        state: FSMContext,
        prompt_source: str | Sequence[str],
    ) -> None:
        prompt_text = _render_text(prompt_source)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await state.update_data(
            upload=None,
            current_models=[],
            last_batch=[],
            preload_message_id=None,
            generation_progress_message_id=None,
        )
        await _send_aux_message(
            message,
            state,
            message.answer,
            prompt_text,
        )

    async def _register_result_message(
        state: FSMContext,
        message: Message,
        model: GlassModel,
        *,
        has_more: bool,
        source_message_id: int | None = None,
    ) -> None:
        data = await state.get_data()
        stored = dict(data.get("result_messages", {}))
        entry = {
            "model_title": model.title,
            "has_more": has_more,
        }
        stored[str(message.message_id)] = entry
        if source_message_id is not None:
            stored[str(source_message_id)] = entry
        await state.update_data(result_messages=stored)

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

    def _map_gender_label(value: str | None) -> str:
        mapping = {
            "male": "Мужской",
            "female": "Женский",
            "unisex": "Унисекс",
            "for_who_male": "Мужской",
            "for_who_female": "Женский",
            "for_who_unisex": "Унисекс",
        }
        if not value:
            return "Унисекс"
        return mapping.get(value, value)

    async def _export_contact_row(
        message: Message,
        phone_number: str,
        gender_value: str | None,
    ) -> None:
        if not phone_number:
            return
        user = message.from_user
        first_name = getattr(user, "first_name", "") or ""
        username = getattr(user, "username", None)
        if username:
            link = f"https://t.me/{username}"
        else:
            link = f"tg://user?id={user.id}"
        record = ContactRecord(
            first_name=first_name,
            phone_number=phone_number,
            telegram_link=link,
            gender=_map_gender_label(gender_value),
        )
        await contact_exporter.export_contact(record)

    async def _store_contact(
        message: Message,
        state: FSMContext,
        phone_e164: str,
        *,
        source: str,
        original_phone: str | None = None,
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
        state_data = await state.get_data()
        await _export_contact_row(
            message,
            original_phone or phone_e164,
            state_data.get("gender"),
        )
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
                track=False,
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
                    generation_progress_message_id=None,
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
        promo_video_path = Path("video") / "promo_start.mp4"
        start_message_sent = False
        if promo_video_path.exists():
            try:
                await _send_aux_message(
                    message,
                    state,
                    message.answer_video,
                    video=FSInputFile(promo_video_path),
                    caption=msg.START_WELCOME,
                    reply_markup=start_keyboard(),
                )
                start_message_sent = True
            except TelegramBadRequest as exc:
                logger.error(
                    "Failed to send promo video from %s: %s", promo_video_path, exc
                )
        else:
            logger.warning("Promo video not found at %s", promo_video_path)

        if not start_message_sent:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.START_WELCOME,
                reply_markup=start_keyboard(),
            )
        await _send_delivery_message(
            message,
            state,
            message.answer,
            msg.MAIN_MENU_HINT,
            reply_markup=main_reply_keyboard(),
        )
        logger.info("%s User %s entered start", EVENT_ID["START"], message.from_user.id)

    @router.callback_query(StateFilter(TryOnStates.START), F.data == "start_go")
    async def start_go(callback: CallbackQuery, state: FSMContext) -> None:
        await state.set_state(TryOnStates.FOR_WHO)
        if callback.message.text:
            await callback.message.edit_text(
                msg.START_GENDER_PROMPT, reply_markup=gender_keyboard()
            )
        else:
            await callback.message.edit_caption(
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
        path = await save_user_photo(message)
        await state.update_data(upload=path, upload_file_id=photo.file_id)
        profile = await repository.ensure_daily_reset(user_id)
        if profile.tries_used == 0:
            await state.update_data(first_generated_today=True)
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await _send_aux_message(
                message,
                state,
                message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            logger.info("%s Limit reached for user %s", EVENT_ID["LIMIT_REACHED"], user_id)
            return
        await _delete_idle_nudge_message(state, message.bot, message.chat.id)
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

    @router.callback_query(StateFilter(TryOnStates.AWAITING_PHOTO), F.data == "send_new_photo")
    async def request_new_photo(callback: CallbackQuery, state: FSMContext) -> None:
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.PHOTO_INSTRUCTION,
        )
        await callback.answer()

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
                _render_text(msg.DAILY_LIMIT_MESSAGE),
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
        await _store_contact(
            message,
            state,
            normalized,
            source="share_button",
            original_phone=contact.phone_number,
        )

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
        upload_file_id = data.get("upload_file_id")

        progress_message: Message | None = None
        user_photo_path: Path | None = None
        result_bytes: bytes | None = None
        start_time = 0.0

        async def _edit_progress(text: str) -> None:
            nonlocal progress_message
            if not progress_message:
                return
            try:
                await progress_message.edit_text(text)
            except TelegramBadRequest as exc:
                logger.debug(
                    "Failed to edit progress message %s for %s: %s",
                    progress_message.message_id,
                    user_id,
                    exc,
                )
                progress_message = None

        try:
            if upload_value and Path(upload_value).exists():
                user_photo_path = Path(upload_value)
            elif upload_file_id:
                downloaded = await redownload_user_photo(
                    message.bot, upload_file_id, user_id
                )
                await state.update_data(upload=downloaded)
                user_photo_path = Path(downloaded)
            else:
                raise RuntimeError("User photo is not available")

            progress_message = await _send_aux_message(
                message,
                state,
                message.answer,
                msg.PROGRESS_DOWNLOADING_USER_PHOTO,
            )
            await state.update_data(
                generation_progress_message_id=progress_message.message_id
            )

            await asyncio.to_thread(resize_inplace, user_photo_path)
            await _edit_progress(msg.PROGRESS_DOWNLOADING_GLASSES)

            if not model.img_nano_url:
                raise RuntimeError("Frame model does not have NanoBanana reference")
            glasses_path = await fetch_drive_file(model.img_nano_url)
            await asyncio.to_thread(resize_inplace, glasses_path)

            await _edit_progress(msg.PROGRESS_SENDING_TO_GENERATION)
            await _edit_progress(msg.PROGRESS_WAIT_GENERATION)

            start_time = time.perf_counter()
            generation_result = await with_generation_slot(
                generate_glasses(
                    face_path=str(user_photo_path),
                    glasses_path=glasses_path,
                )
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            result_bytes = generation_result.image_bytes
            result_kb = len(result_bytes) / 1024 if result_bytes else 0
            logger.info(
                (
                    "NanoBanana generation: user_id=%s frame_id=%s latency_ms=%s "
                    "result_kb=%.1f finish_reason=%s attempt=%s retried=%s"
                ),
                user_id,
                model.unique_id,
                latency_ms,
                result_kb,
                generation_result.finish_reason,
                generation_result.attempt,
                generation_result.retried,
            )

        except NanoBananaGenerationError as exc:
            latency_ms = (
                int((time.perf_counter() - start_time) * 1000)
                if start_time
                else 0
            )
            logger.warning(
                (
                    "NanoBanana failure: user_id=%s frame_id=%s finish_reason=%s "
                    "latency_ms=%s has_inline=%s has_data_url=%s has_file_uri=%s "
                    "detail=%s"
                ),
                user_id,
                model.unique_id,
                exc.finish_reason,
                latency_ms,
                exc.has_inline,
                exc.has_data_url,
                exc.has_file_uri,
                exc.reason_detail,
            )
            await _delete_state_message(message, state, "generation_progress_message_id")
            await state.update_data(
                selected_model=None,
                current_models=[],
                upload=None,
                upload_file_id=None,
            )
            await state.set_state(TryOnStates.AWAITING_PHOTO)
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.PHOTO_NOT_SUITABLE_MAIN,
                reply_markup=send_new_photo_keyboard(),
            )
            return
        except Exception as exc:  # noqa: BLE001
            latency_ms = (
                int((time.perf_counter() - start_time) * 1000)
                if start_time
                else 0
            )
            logger.error(
                "%s Generation failed: %s (latency_ms=%s)",
                EVENT_ID["GENERATION_FAILED"],
                exc,
                latency_ms,
                exc_info=True,
            )
            logger.warning(
                (
                    "NanoBanana failure: user_id=%s frame_id=%s finish_reason=%s "
                    "latency_ms=%s has_inline=%s has_data_url=%s has_file_uri=%s "
                    "detail=%s"
                ),
                user_id,
                model.unique_id,
                None,
                latency_ms,
                False,
                False,
                False,
                str(exc),
            )
            await _delete_state_message(message, state, "generation_progress_message_id")
            await state.update_data(
                selected_model=None,
                current_models=[],
                upload=None,
                upload_file_id=None,
            )
            await state.set_state(TryOnStates.AWAITING_PHOTO)
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.PHOTO_NOT_SUITABLE_MAIN,
                reply_markup=send_new_photo_keyboard(),
            )
            return
        finally:
            if user_photo_path and user_photo_path.exists():
                try:
                    user_photo_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Failed to remove temp file %s", user_photo_path)
            await state.update_data(upload=None)

        await repository.inc_used_on_success(user_id)
        remaining = await repository.remaining_tries(user_id)
        plan = resolve_generation_followup(
            first_generated_today=data.get("first_generated_today", True),
            remaining=remaining,
        )
        gen_count_before = await repository.get_generation_count(user_id)
        is_second_generation = gen_count_before == 1
        caption_source = (
            msg.FIRST_RESULT_CAPTION
            if data.get("first_generated_today", True)
            else msg.NEXT_RESULT_CAPTION
        )
        caption_text = _render_text(caption_source)
        result_has_more = plan.remaining > 0
        keyboard_remaining = plan.remaining if result_has_more else 0
        if plan.outcome is GenerationOutcome.LIMIT:
            caption_text = model.title
            result_has_more = False
            keyboard_remaining = 0
        elif is_second_generation:
            caption_text = model.title
            result_has_more = False
            keyboard_remaining = 0
        result_markup = generation_result_keyboard(
            model.site_url, keyboard_remaining
        )
        await _delete_state_message(message, state, "generation_progress_message_id")
        result_message = await _send_delivery_message(
            message,
            state,
            message.answer_photo,
            BufferedInputFile(result_bytes, filename="result.png"),
            caption=caption_text,
            reply_markup=result_markup,
        )
        await _register_result_message(
            state,
            result_message,
            model,
            has_more=result_has_more,
            source_message_id=message.message_id,
        )
        new_gen_count = await repository.increment_generation_count(user_id)
        new_flag = next_first_flag_value(
            data.get("first_generated_today", True), plan.outcome
        )
        await state.update_data(first_generated_today=new_flag)
        contact_data = await state.get_data()
        contact_active_before = contact_data.get("contact_request_active", False)
        if plan.outcome is GenerationOutcome.LIMIT:
            limit_text = _render_text(msg.DAILY_LIMIT_MESSAGE)
            await _send_delivery_message(
                message,
                state,
                message.answer,
                limit_text,
                reply_markup=limit_reached_keyboard(site_url),
            )
            if contact_active_before:
                await state.update_data(contact_pending_result_state="limit")
            else:
                await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            logger.info(
                "%s Limit reached post generation %s", EVENT_ID["LIMIT_REACHED"], user_id
            )
        else:
            contact_requested_now = False
            if (
                not contact_active_before
                and new_gen_count >= CONTACT_INITIAL_TRIGGER
                and (is_second_generation or new_gen_count == CONTACT_INITIAL_TRIGGER)
            ):
                contact_requested_now = await _maybe_request_contact(
                    message,
                    state,
                    user_id,
                    origin_state=TryOnStates.RESULT.state,
                )
                if contact_requested_now:
                    logger.info(
                        "%s Contact requested post generation %s",
                        EVENT_ID["MODELS_SENT"],
                        user_id,
                    )

            if contact_active_before and not contact_requested_now:
                await state.update_data(contact_pending_result_state="result")
            elif not contact_requested_now:
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
            data = await state.get_data()
            stored_results = dict(data.get("result_messages", {}))
            entry = stored_results.get(str(message.message_id))
            if entry:
                target_markup = updated_markup if updated_markup is not None else current_markup
                try:
                    await message.edit_caption(
                        caption=entry.get("model_title", ""),
                        reply_markup=target_markup,
                    )
                except TelegramBadRequest as exc:
                    logger.debug(
                        "Failed to update caption for message %s: %s",
                        message.message_id,
                        exc,
                    )
                else:
                    entry["has_more"] = False
                    stored_results[str(message.message_id)] = entry
                    await state.update_data(result_messages=stored_results)
        chat_id = message.chat.id if message else user_id
        await _delete_idle_nudge_message(state, callback.bot, chat_id)
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await _send_aux_message(
                callback.message,
                state,
                callback.message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            await callback.answer()
            return
        if message:
            await _prompt_for_next_photo(message, state, msg.PHOTO_INSTRUCTION)
        else:
            await state.set_state(TryOnStates.AWAITING_PHOTO)
        if remove_source_message and message:
            try:
                await message.bot.delete_message(message.chat.id, message.message_id)
            except TelegramBadRequest as exc:
                logger.debug(
                    "Failed to delete reminder message %s: %s",
                    message.message_id,
                    exc,
                )
        await callback.answer()

    @router.message(F.text == msg.MAIN_MENU_TRY_BUTTON)
    async def handle_main_menu_try(message: Message, state: FSMContext) -> None:
        current_state = await state.get_state()
        if current_state == ContactRequest.waiting_for_phone.state:
            return
        user_id = message.from_user.id
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await _send_aux_message(
                message,
                state,
                message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            return
        gen_count = await repository.get_generation_count(user_id)
        prompt_source = (
            msg.PHOTO_INSTRUCTION if gen_count == 0 else msg.NEXT_RESULT_CAPTION
        )
        await _prompt_for_next_photo(message, state, prompt_source)

    @router.message(F.text == msg.MAIN_MENU_POLICY_BUTTON)
    async def handle_main_menu_policy(message: Message, state: FSMContext) -> None:
        await _send_privacy_policy(message, state)

    @router.message(Command("privacy"))
    async def command_privacy(message: Message, state: FSMContext) -> None:
        await _send_privacy_policy(message, state)

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
        first_flag = profile.tries_used == 0
        await state.update_data(gender=profile.gender, first_generated_today=first_flag)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.PHOTO_INSTRUCTION,
        )
        await callback.answer()

    return router
