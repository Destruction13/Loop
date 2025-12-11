
"""FSM definitions and handler registration."""

from __future__ import annotations

import asyncio
from email import message
import io
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Mapping, Optional, Sequence
from aiogram import BaseMiddleware, F, Router, Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
)
from aiogram.types.input_file import BufferedInputFile, FSInputFile, URLInputFile

from app.analytics import track_event
from app.keyboards import (
    CONTACT_NEVER_CALLBACK,
    CONTACT_SHARE_CALLBACK,
    CONTACT_SKIP_CALLBACK,
    REUSE_SAME_PHOTO_CALLBACK,
    batch_selection_keyboard,
    contact_request_keyboard,
    contact_share_reply_keyboard,
    gender_keyboard,
    generation_result_keyboard,
    idle_reminder_keyboard,
    limit_reached_keyboard,
    more_buttonless_markup,
    privacy_policy_keyboard,
    promo_keyboard,
    remove_more_button,
    reuse_same_photo_keyboard,
    send_new_photo_keyboard,
    start_keyboard,
)
from app.models import FilterOptions, GlassModel, UserContact
from app.services.catalog_base import CatalogError
from app.config import CollageConfig
from app.media import probe_video_size
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
from logger import get_logger, info_domain


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
CONTACT_REMINDER_TRIGGER = 6


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
    promo_video_path: Path,
    promo_video_enabled: bool,
    promo_video_width: int | None,
    promo_video_height: int | None,
) -> Router:
    router = Router()
    logger = get_logger("bot.handlers")

    idle_delay = max(int(idle_nudge_seconds), 0)
    idle_enabled = enable_idle_nudge and idle_delay > 0
    idle_tasks: dict[int, asyncio.Task] = {}

    policy_url = (privacy_policy_url or "").strip()
    policy_button_url = (
        policy_url or "https://telegra.ph/Politika-konfidencialnosti-LOOV-10-29"
    )
    BUSY_STATES = {TryOnStates.GENERATING.state, TryOnStates.SHOW_RECS.state}
    INVISIBLE_PROMPT = "\u2060"

    def _detect_card_mode(message: Message) -> str:
        content_type = getattr(message, "content_type", "")
        return "text" if content_type == "text" else "caption"

    def _resolve_chat_id(message: Message | None) -> int | None:
        if not message:
            return None
        chat = getattr(message, "chat", None)
        if chat is not None and hasattr(chat, "id"):
            try:
                return int(chat.id)
            except (TypeError, ValueError):
                return None
        chat_id = getattr(message, "chat_id", None)
        if chat_id is not None:
            try:
                return int(chat_id)
            except (TypeError, ValueError):
                return None
        from_user = getattr(message, "from_user", None)
        if from_user is not None and hasattr(from_user, "id"):
            try:
                return int(from_user.id)
            except (TypeError, ValueError):
                return None
        return None

    async def _remember_card_message(
        state: FSMContext,
        message: Message | None,
        *,
        title: str | None,
        trimmed: bool = False,
    ) -> None:
        if not message:
            return
        chat_id = _resolve_chat_id(message)
        if chat_id is None:
            return
        entry = {
            "message_id": int(message.message_id),
            "chat_id": chat_id,
            "type": _detect_card_mode(message),
            "title": title,
            "trimmed": trimmed,
            "trim_failed": False,
        }
        await state.update_data(last_card_message=entry)

    from app.keyboards import generation_result_keyboard  # добавь вверху, если ещё не импортировано

    async def _trim_last_card_message(
        message: Message | None,
        state: FSMContext,
        *,
        site_url: str,  # <---- вот так пробрасываем
        title: str | None = None,
    ) -> None:
        if not message:
            return
        data = await state.get_data()
        entry = dict(data.get("last_card_message") or {})
        if not entry:
            return
        message_id = entry.get("message_id")
        if not message_id:
            return
        if entry.get("trim_failed") and title is None:
            return
        stored_title = entry.get("title")
        if entry.get("trimmed") and title is None:
            return
        if entry.get("trimmed") and title is not None and stored_title == title:
            return
        final_title = title or stored_title
        if not final_title:
            return
        chat_id = entry.get("chat_id") or _resolve_chat_id(message)
        if chat_id is None:
            return
        mode = entry.get("type") or "caption"
        if mode == "text":
            return
        bot = message.bot
        

        # Клавиатура только с кнопкой «Подробнее»
        current_markup = generation_result_keyboard(site_url, remaining=0)

        try:
            if mode == "text":
                await bot.edit_message_text(
                    f"<b>{final_title}</b>",
                    chat_id=chat_id,
                    message_id=int(message_id),
                    reply_markup=current_markup, parse_mode=ParseMode.HTML
                )
            else:
                await bot.edit_message_caption(
                    chat_id=chat_id,
                    message_id=int(message_id),
                    caption=f"<b>{final_title}</b>",
                    reply_markup=current_markup, parse_mode=ParseMode.HTML
                )
        except (TelegramBadRequest, TelegramForbiddenError, AttributeError) as exc:
            logger.debug(
                "Failed to trim card message %s: %s",
                message_id,
                exc,
                extra={"stage": "CARD_TRIM"},
            )
            entry["trim_failed"] = True
            await state.update_data(last_card_message=entry)
        else:
            entry.update(
                {
                    "title": final_title,
                    "trimmed": True,
                    "trim_failed": False,
                    "type": mode,
                    "chat_id": chat_id,
                }
            )
            await state.update_data(last_card_message=entry)


    async def _trim_message_card(
        message: Message | None,
        state: FSMContext,
        *,
        title: str,
    ) -> None:
        if not message:
            return
        mode = _detect_card_mode(message)
        if mode == "text":
            return
        try:
            if mode == "text":
                await message.edit_text(f"<b>{title}</b>", reply_markup=message.reply_markup, parse_mode=ParseMode.HTML)
            else:
                await message.edit_caption(caption=f"<b>{title}</b>", reply_markup=message.reply_markup, parse_mode=ParseMode.HTML)
        except TelegramBadRequest as exc:
            logger.debug(
                "Failed to trim inline card %s: %s",
                message.message_id,
                exc,
                extra={"stage": "CARD_TRIM"},
            )
            # keep markup
            chat_id = _resolve_chat_id(message)
            if chat_id is None:
                return
            entry = {
                "message_id": int(message.message_id),
                "chat_id": chat_id,
                "type": mode,
                "title": title,
                "trimmed": False,
                "trim_failed": True,
            }
            await state.update_data(last_card_message=entry)
        else:
            chat_id = _resolve_chat_id(message)
            if chat_id is None:
                return
            entry = {
                "message_id": int(message.message_id),
                "chat_id": chat_id,
                "type": mode,
                "title": title,
                "trimmed": True,
                "trim_failed": False,
            }
            await state.update_data(last_card_message=entry)

    async def _dismiss_reply_keyboard(message: Message | None) -> None:
        if not message:
            return
        try:
            # отправляем невидимый символ с удалением reply-клавиатуры
            tmp = await message.answer(INVISIBLE_PROMPT, reply_markup=ReplyKeyboardRemove())
            # сразу же удаляем техсообщение
            try:
                await message.bot.delete_message(message.chat.id, tmp.message_id)
            except TelegramBadRequest:
                pass
        except TelegramBadRequest as exc:
            logger.debug(
                "Failed to hide reply keyboard for %s: %s",
                message.message_id,
                exc,
                extra={"stage": "CONTACT_KEYBOARD_REMOVE"},
            )


    async def _is_generation_in_progress(state: FSMContext) -> bool:
        data = await state.get_data()
        if data.get("is_generating"):
            return True
        current_state = await state.get_state()
        return current_state in BUSY_STATES

    async def _is_command_locked(
        state: FSMContext, *, allow_show_recs: bool = False
    ) -> bool:
        data = await state.get_data()
        current_state = await state.get_state()

        # 1) Жёстко занято, если флаг поднят
        if data.get("is_generating"):
            return True

        # 2) Страховка от гонки сразу после приёма фото
        #    (фото уже есть, стейт ещё None или только SHOW_RECS)
        if data.get("upload") or data.get("upload_file_id"):
            if current_state is None or current_state == TryOnStates.SHOW_RECS.state:
                return True

        # 3) Любой busy-стейт — занято
        if current_state in BUSY_STATES:
            return True

        # 4) Пока ждём фото — занято, если есть незавершённая загрузка/генерация
        if current_state == TryOnStates.AWAITING_PHOTO.state:
            return bool(
                data.get("upload")
                or data.get("upload_file_id")
                or data.get("generation_progress_message_id")
            )
        return False

    # Try-on cycles:
    # /start and /wear always bump current_cycle so a fresh flow can start while older generations finish in the background.
    # Each generation remembers the cycle it was launched with; stale cycles are delivered without the "try more" button,
    # while the active cycle keeps the full keyboard and state transitions.
    # A fresh photo upload also spins up its own cycle so older collages remain clickable but their generations are treated as stale.
    async def _ensure_current_cycle_id(state: FSMContext, user_id: int) -> int:
        """Return the active try-on cycle marker stored in FSM or repository."""

        data = await state.get_data()
        raw_cycle = data.get("current_cycle")
        if raw_cycle is not None:
            try:
                return int(raw_cycle)
            except (TypeError, ValueError):
                pass
        profile = await repository.ensure_user(user_id)
        current_cycle = getattr(profile, "cycle_index", 0) or 0
        await state.update_data(current_cycle=current_cycle)
        return current_cycle

    async def _start_new_cycle(state: FSMContext, user_id: int) -> int:
        """Increment cycle index so older generations become stale."""

        current_cycle = await repository.start_new_tryon_cycle(user_id)
        await state.update_data(current_cycle=current_cycle)
        return current_cycle

    async def _is_cycle_current(state: FSMContext, cycle_id: int) -> bool:
        data = await state.get_data()
        try:
            return int(data.get("current_cycle")) == int(cycle_id)
        except (TypeError, ValueError):
            return False

    async def _cleanup_cycle_messages(
        message: Message,
        state: FSMContext,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """Remove stored messages of the previous cycle without failing the flow."""

        snapshot = dict(data or await state.get_data())
        updates: dict[str, Any] = {}
        message_keys = (
            "preload_message_id",
            "generation_progress_message_id",
            "models_message_id",
            "gender_prompt_message_id",
        )
        for key in message_keys:
            mid = snapshot.get(key)
            if not mid:
                continue
            try:
                await message.bot.delete_message(message.chat.id, int(mid))
            except (TelegramBadRequest, TelegramForbiddenError) as exc:
                logger.debug("Failed to delete %s %s: %s", key, mid, exc)
            updates[key] = None
        prompt_id = snapshot.get("contact_prompt_message_id")
        if prompt_id:
            try:
                await message.bot.delete_message(message.chat.id, int(prompt_id))
            except (TelegramBadRequest, TelegramForbiddenError) as exc:
                logger.debug("Failed to delete contact prompt %s: %s", prompt_id, exc)
            updates["contact_prompt_message_id"] = None
        busy_ids = list(snapshot.get("busy_message_ids") or [])
        if busy_ids:
            for mid in busy_ids:
                try:
                    await message.bot.delete_message(message.chat.id, int(mid))
                except (TelegramBadRequest, TelegramForbiddenError) as exc:
                    logger.debug("Failed to delete busy message %s: %s", mid, exc)
            updates["busy_message_ids"] = []
        if updates:
            await state.update_data(**updates)


    async def _reject_if_busy(
        message: Message,
        state: FSMContext,
        *,
        allow_show_recs: bool = False,
        busy_message: str | None = None,
    ) -> bool:
        # если команда заблокирована — отвечаем busy и запоминаем id сообщения
        if await _is_command_locked(state, allow_show_recs=allow_show_recs):
            text = busy_message or msg.GENERATION_BUSY
            sent = await message.answer(text)
            data = await state.get_data()
            ids = list(data.get("busy_message_ids", []))
            ids.append(sent.message_id)
            await state.update_data(busy_message_ids=ids)
            return True
        return False


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
        last_activity_ts = getattr(profile, "last_activity_ts", 0) or 0
        if last_activity_ts:
            seconds_since = now.timestamp() - float(last_activity_ts)
            if seconds_since < 300:
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
                await _deactivate_previous_more_button(bot, user_id)
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
            await repository.set_last_more_message(
                user_id,
                message.message_id,
                "idle",
                {"site_url": site_url},
            )
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

    async def _delete_busy_messages(state: FSMContext, bot: Bot, chat_id: int) -> None:
        data = await state.get_data()
        ids = list(data.get("busy_message_ids") or [])
        if not ids:
            return
        for mid in ids:
            try:
                await bot.delete_message(chat_id, int(mid))
            except (TelegramBadRequest, TelegramForbiddenError) as exc:
                logger.debug("Failed to delete busy message %s: %s", mid, exc)
        await state.update_data(busy_message_ids=[])

    async def _edit_last_aux_message(
        message: Message,
        state: FSMContext,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        data = await state.get_data()
        message_id = data.get("last_aux_message_id")
        if not message_id:
            return False
        try:
            await message.bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=int(message_id),
                reply_markup=reply_markup,
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            return False
        else:
            await state.update_data(last_aux_message_id=int(message_id))
            return True

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
        if "reply_markup" not in kwargs:
            kwargs["reply_markup"] = None
        sent_message = await send_method(*send_args, **kwargs)
        if track:
            await state.update_data(last_aux_message_id=sent_message.message_id)
        elif delete_previous:
            await state.update_data(last_aux_message_id=None)
        return sent_message

    async def _delete_phone_invalid_message(
        message: Message,
        state: FSMContext,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        payload = data or await state.get_data()
        invalid_id = payload.get("phone_invalid_message_id")
        if not invalid_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, int(invalid_id))
        except TelegramBadRequest as exc:
            logger.debug(
                "Failed to delete invalid phone message %s: %s",
                invalid_id,
                exc,
            )
        finally:
            await state.update_data(phone_invalid_message_id=None)

    async def _reset_phone_attempts(message: Message, state: FSMContext) -> None:
        await _delete_phone_invalid_message(message, state)
        await state.update_data(phone_bad_attempts=0)

    async def _send_delivery_message(
        source_message: Message,
        state: FSMContext,
        send_method: Callable[..., Awaitable[Message]],
        *args,
        **kwargs,
    ) -> Message:
        return await send_method(*args, **kwargs)

    async def _deactivate_previous_more_button(bot: Bot, user_id: int) -> None:
        profile = await repository.ensure_user(user_id)
        message_id = profile.last_more_message_id
        message_type = profile.last_more_message_type
        if not message_id or not message_type:
            return
        markup = more_buttonless_markup(message_type, profile.last_more_message_payload)
        if markup is None:
            await repository.set_last_more_message(user_id, None, None, None)
            return
        try:
            await bot.edit_message_reply_markup(
                chat_id=user_id, message_id=message_id, reply_markup=markup
            )
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            logger.debug(
                "Failed to update previous more button %s for %s: %s",
                message_id,
                user_id,
                exc,
            )
        finally:
            await repository.set_last_more_message(user_id, None, None, None)

    async def _clear_reuse_offer(state: FSMContext, bot: Bot, chat_id: int) -> None:
        data = await state.get_data()
        message_id = data.get("reuse_offer_message_id")
        if not message_id:
            if data.get("reuse_offer_active"):
                await state.update_data(reuse_offer_active=False)
            return
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=int(message_id), reply_markup=None
            )
        except TelegramBadRequest as exc:
            logger.debug(
                "Failed to clear reuse offer %s: %s",
                message_id,
                exc,
                extra={"stage": "REUSE_CLEAR"},
            )
        finally:
            await state.update_data(
                reuse_offer_message_id=None,
                reuse_offer_active=False,
            )

    def _render_text(source: str | Sequence[str]) -> str:
        if isinstance(source, (list, tuple)):
            return "".join(source)
        return str(source)

    FOLLOWUP_CAPTIONS: tuple[Sequence[str] | str, ...] = (
        msg.SECOND_RESULT_CAPTION,
        msg.THIRD_RESULT_CAPTION,
        msg.FOURTH_RESULT_CAPTION_TEMPLATE,
        msg.FIFTH_RESULT_CAPTION,
        msg.SIXTH_RESULT_CAPTION,
        msg.SEVENTH_RESULT_CAPTION,
    )

    def _resolve_ready_word(gender: str | None) -> str:
        mapping = {
            "male": "Готов",
            "for_who_male": "Готов",
            "female": "Готова",
            "for_who_female": "Готова",
        }
        return mapping.get(gender, "Готов(а)")

    def _resolve_followup_caption(index: int, gender: str | None) -> str:
        if not FOLLOWUP_CAPTIONS:
            return _render_text(msg.SECOND_RESULT_CAPTION)
        normalized = index % len(FOLLOWUP_CAPTIONS)
        if normalized == 2:
            template_source = msg.FOURTH_RESULT_CAPTION_TEMPLATE
            template_text = (
                "".join(template_source)
                if isinstance(template_source, (list, tuple))
                else str(template_source)
            )
            return template_text.format(ready=_resolve_ready_word(gender))
        source = FOLLOWUP_CAPTIONS[normalized]
        return _render_text(source)

    def _compose_result_caption(model: GlassModel, body: str) -> str:
        model_name = getattr(model, "name", None) or model.title
        title_line = f"<b>{model_name}</b>"
        stripped = body.strip()
        if not stripped:
            return title_line
        return f"{title_line}\n\n{stripped}"

    async def _prompt_for_next_photo(
        message: Message,
        state: FSMContext,
        prompt_source: str | Sequence[str],
        *,
        cycle_id: int | None = None,
    ) -> None:
        prompt_text = _render_text(prompt_source)
        await _deactivate_previous_more_button(message.bot, message.chat.id)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        resolved_cycle = cycle_id
        if resolved_cycle is None:
            resolved_cycle = await _ensure_current_cycle_id(state, message.from_user.id)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await state.update_data(
            upload=None,
            current_models=[],
            last_batch=[],
            preload_message_id=None,
            generation_progress_message_id=None,
            presented_model_ids=[],
            is_generating=False,
            contact_prompt_due=None,
            suppress_more_button=False,
            reuse_offer_message_id=None,
            reuse_offer_active=False,
            allow_more_button_next=False,
            current_cycle=resolved_cycle,
        )
        await repository.set_last_more_message(message.chat.id, None, None, None)
        await _send_aux_message(
            message,
            state,
            message.answer,
            prompt_text,
        )

    async def _send_reuse_prompt(
        message: Message,
        state: FSMContext,
        prompt_source: str | Sequence[str],
    ) -> None:
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        prompt_text = _render_text(prompt_source)
        sent_message = await _send_aux_message(
            message,
            state,
            message.answer,
            prompt_text,
            reply_markup=reuse_same_photo_keyboard(),
            delete_previous=False,
        )
        await state.update_data(
            reuse_offer_message_id=sent_message.message_id,
            reuse_offer_active=True,
            suppress_more_button=True,
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

        # ВАЖНО: для другого message_id — отдельный dict, НЕ та же ссылка
        if source_message_id is not None:
            stored[str(source_message_id)] = dict(entry)

        await state.update_data(result_messages=stored)
        await _remember_card_message(state, message, title=model.title, trimmed=False)


    async def _maybe_request_contact(
        message: Message,
        state: FSMContext,
        user_id: int,
        *,
        origin_state: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> bool:
        data = await state.get_data()
        if data.get("contact_request_active"):
            return True
        effective_trigger = trigger or data.get("contact_prompt_due")
        if not effective_trigger:
            return False
        cooldown = max(int(data.get("contact_request_cooldown") or 0), 0)
        if cooldown > 0:
            logger.debug(
                "Contact prompt cooldown active for user %s (remaining=%s)",
                user_id,
                cooldown,
            )
            return False
        profile = await repository.ensure_user(user_id)
        if profile.contact_skip_once:
            if cooldown > 0:
                return False
            await repository.set_contact_skip_once(user_id, False)
        if profile.contact_never:
            await state.update_data(contact_prompt_due=None)
            await repository.mark_contact_prompt_sent(user_id, effective_trigger)
            return False
        contact = await repository.get_user_contact(user_id)
        if contact and contact.consent:
            await state.update_data(contact_prompt_due=None)
            await repository.mark_contact_prompt_sent(user_id, effective_trigger)
            return False
        current_state = origin_state or await state.get_state()
        pending_state = data.get("contact_pending_result_state")
        if not pending_state and current_state == TryOnStates.RESULT.state:
            pending_state = "result"
        await _trim_last_card_message(message, state, site_url=site_url)
        prompt_text = (
            f"<b>{msg.ASK_PHONE_TITLE}</b>\n\n"
            f"{msg.ASK_PHONE_BODY.format(rub=contact_reward_rub)}"
        )
        update_payload = {
            "contact_request_active": True,
            "contact_pending_generation": True,
        }
        if pending_state and pending_state != data.get("contact_pending_result_state"):
            update_payload["contact_pending_result_state"] = pending_state
        await _deactivate_previous_more_button(message.bot, user_id)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        await repository.set_last_more_message(user_id, None, None, None)
        markup = contact_request_keyboard()
        prompt_message_id: int | None = None
        if await _edit_last_aux_message(
            message, state, prompt_text, reply_markup=markup
        ):
            refreshed = await state.get_data()
            stored_id = refreshed.get("last_aux_message_id")
            if stored_id:
                prompt_message_id = int(stored_id)
        if prompt_message_id is None:
            prompt_message = await _send_aux_message(
                message,
                state,
                message.answer,
                prompt_text,
                reply_markup=markup,
            )
            prompt_message_id = prompt_message.message_id
        else:
            await state.update_data(last_aux_message_id=prompt_message_id)
        update_payload.update(
            phone_bad_attempts=0,
            phone_invalid_message_id=None,
            contact_prompt_message_id=prompt_message_id,
            contact_request_cooldown=0,
        )
        await state.update_data(
            **update_payload, contact_prompt_due=None
        )
        await repository.mark_contact_prompt_sent(user_id, effective_trigger)
        await state.set_state(ContactRequest.waiting_for_phone)
        logger.debug("Contact request issued for user %s", user_id)
        info_domain(
            "bot.handlers",
            "⛔ Генерация пропущена — причина=contact_request",
            stage="GENERATION_SKIPPED",
            user_id=user_id,
            pending_state=pending_state or current_state,
        )
        return True

    async def _send_models(
        message: Message,
        user_id: int,
        filters: FilterOptions,
        state: FSMContext,
        *,
        skip_contact_prompt: bool = False,
        exclude_ids: set[str] | None = None,
        cycle_id: int | None = None,
        photo_context: dict[str, Any] | None = None,
    ) -> bool:
        """Send model suggestions for a specific try-on cycle/photo context snapshot."""

        await _trim_last_card_message(message, state, site_url=site_url)
        if not skip_contact_prompt:
            if await _maybe_request_contact(message, state, user_id):
                return False
        if cycle_id is None:
            cycle_id = await _ensure_current_cycle_id(state, user_id)
        data = await state.get_data()
        effective_photo_context = dict(photo_context or {})
        if not effective_photo_context:
            effective_photo_context = {
                "upload": data.get("upload"),
                "upload_file_id": data.get("upload_file_id"),
                "last_photo_file_id": data.get("last_photo_file_id"),
            }
        try:
            result = await recommender.recommend_for_user(
                user_id,
                filters.gender,
                exclude_ids=exclude_ids or set(),
            )
        except CatalogError as exc:
            logger.error(
                "Ошибка при получении каталога: %s",
                exc,
                extra={"stage": "SHEET_PARSE_ERROR"},
            )
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
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.CATALOG_TEMPORARILY_UNAVAILABLE,
            )
            await state.update_data(current_models=[], last_batch=[])
            await _delete_state_message(message, state, "preload_message_id")
            info_domain(
                "bot.handlers",
                "⛔ Генерация пропущена — причина=no_models",
                stage="GENERATION_SKIPPED",
                user_id=user_id,
                gender=filters.gender,
            )
            return False
        batch = list(result.models)
        data = await state.get_data()
        presented = list(dict.fromkeys(data.get("presented_model_ids", [])))
        for model in batch:
            if model.unique_id not in presented:
                presented.append(model.unique_id)
        await state.update_data(
            current_models=batch,
            last_batch=batch,
            presented_model_ids=presented,
        )
        await _send_model_batches(
            message,
            state,
            batch,
            cycle_id=cycle_id,
            photo_context=effective_photo_context,
        )
        await _delete_state_message(message, state, "preload_message_id")
        await state.update_data(is_generating=False)
        if result.exhausted:
            # Карточка «Ты просмотрел все модели» отключена продуктовой командой.
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.CATALOG_TEMPORARILY_UNAVAILABLE,
            )
        return True

    async def _send_model_batches(
        message: Message,
        state: FSMContext,
        batch: list[GlassModel],
        *,
        cycle_id: int,
        photo_context: dict[str, Any],
    ) -> None:
        groups = chunk_models(batch, batch_size)
        for group in groups:
            try:
                await _send_batch_message(
                    message,
                    state,
                    group,
                    cycle_id=cycle_id,
                    photo_context=photo_context,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Unexpected collage error for models %s: %s",
                    [model.unique_id for model in group],
                    exc,
                    extra={"stage": "MODELS_SENT"},
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
            contact_prompt_message_id=None,
            suppress_more_button=False,
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

    async def _complete_contact_skip(
        message: Message,
        state: FSMContext,
        *,
        manual: bool = False,
    ) -> None:
        user_id = message.from_user.id
        await _delete_contact_prompt_message(message, state)
        await _dismiss_reply_keyboard(message)
        await repository.set_contact_skip_once(user_id, True)
        await _reset_phone_attempts(message, state)
        await state.update_data(contact_request_cooldown=4, contact_prompt_due=None, allow_more_button_next=True)
        await _resume_after_contact(message, state, send_generation=False)
        current_state = await state.get_state()
        if current_state != TryOnStates.DAILY_LIMIT_REACHED.state:
            await _send_reuse_prompt(message, state, msg.ASK_PHONE_SKIP_ACK)
        if manual:
            logger.debug("User %s skipped contact once", user_id)
        else:
            logger.debug("User %s auto-skipped phone request", user_id)

    async def _handle_phone_invalid_attempt(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        attempts = int(data.get("phone_bad_attempts") or 0) + 1
        await state.update_data(phone_bad_attempts=attempts)
        await _delete_phone_invalid_message(message, state, data=data)
        if attempts >= 3:
            await _complete_contact_skip(message, state)
            return
        invalid_message = await message.answer(msg.ASK_PHONE_INVALID)
        await state.update_data(phone_invalid_message_id=invalid_message.message_id)

    async def _export_lead(
        user_id: int,
        phone_e164: str,
        source: str,
        consent_ts: int,
        *,
        username: str | None,
        full_name: str | None,
    ) -> bool:
        payload = LeadPayload(
            tg_user_id=user_id,
            phone_e164=phone_e164,
            source=source,
            consent_ts=consent_ts,
            username=username,
            full_name=full_name,
        )
        return await leads_exporter.export_lead_to_sheet(payload)

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
        await _delete_contact_prompt_message(message, state)
        user = message.from_user
        user_id = user.id
        await track_event(str(user_id), "phone_shared", value="yes")
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
        await repository.save_contact(user_id, original_phone or phone_e164)
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
        export_ok = False
        if changed:
            export_ok = await _export_lead(
                user_id,
                phone_e164,
                source,
                consent_ts,
                username=username,
                full_name=full_name,
            )
            if export_ok:
                await track_event(str(user_id), "lead_export_ok")
        if reward_needed:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_THANKS.format(
                    rub=contact_reward_rub, promo=promo_contact_code
                ),
                track=False,
                delete_previous=False,
                reply_markup=ReplyKeyboardRemove(),
            )
            logger.debug(
                "Contact stored for user %s (source=%s)",
                user_id,
                source,
            )
            await state.update_data(
                contact_request_cooldown=0,
                contact_prompt_due=None,
                allow_more_button_next=True,
            )
            await _resume_after_contact(message, state, send_generation=False)
            current_state = await state.get_state()
            if current_state != TryOnStates.DAILY_LIMIT_REACHED.state:
                gender = state_data.get("gender")
                gen_count = await repository.get_generation_count(user_id)
                followup_index = max(gen_count, 1) - 1
                followup_text = _resolve_followup_caption(
                    followup_index,
                    gender,
                )
                await _send_reuse_prompt(message, state, followup_text)
            return
        else:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.ASK_PHONE_ALREADY_HAVE,
                reply_markup=ReplyKeyboardRemove(),
            )
            logger.debug("Contact already existed for user %s", user_id)
        await state.update_data(
            contact_request_cooldown=0,
            contact_prompt_due=None,
            allow_more_button_next=False,
        )
        await _resume_after_contact(message, state, send_generation=True)

    async def _handle_manual_phone(
        message: Message, state: FSMContext, *, source: str
    ) -> None:
        raw = (message.text or "").strip()
        normalized = normalize_phone(raw)
        if not normalized:
            await _handle_phone_invalid_attempt(message, state)
            return
        await _reset_phone_attempts(message, state)
        await _store_contact(message, state, normalized, source=source)

    async def _send_batch_message(
        message: Message,
        state: FSMContext,
        group: tuple[GlassModel, ...],
        *,
        cycle_id: int,
        photo_context: dict[str, Any],
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
            m = await _send_aux_message(
                message,
                state,
                message.answer,
                msg.COLLAGE_IMAGES_UNAVAILABLE,
                reply_markup=keyboard,
            )
            sessions = dict((await state.get_data()).get("collage_sessions", {}))
            sessions[str(m.message_id)] = {
                "models": list(group),
                "cycle": cycle_id,
                "upload": photo_context.get("upload"),
                "upload_file_id": photo_context.get("upload_file_id"),
                "last_photo_file_id": photo_context.get("last_photo_file_id"),
                "aliases": [str(message.message_id)],
            }
            await state.update_data(models_message_id=m.message_id, collage_sessions=sessions)
            return
        except CollageProcessingError as exc:
            logger.warning(
                "Collage processing failed for models %s: %s",
                [model.unique_id for model in group],
                exc,
                extra={"collage_fallback_used": True},
            )
            await _send_batch_as_photos(
                message,
                state,
                group,
                reply_markup=keyboard,
                cycle_id=cycle_id,
                photo_context=photo_context,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected collage error for models %s: %s",
                [model.unique_id for model in group],
                exc,
                exc_info=True,
                extra={"collage_fallback_used": True},
            )
            await _send_batch_as_photos(
                message,
                state,
                group,
                reply_markup=keyboard,
                cycle_id=cycle_id,
                photo_context=photo_context,
            )
            return

        filename = f"collage-{uuid.uuid4().hex}.jpg"
        collage_bytes = buffer.getvalue()
        buffer.close()
        try:
            sent = await _send_delivery_message(
                message,
                state,
                message.answer_photo,
                photo=BufferedInputFile(collage_bytes, filename=filename),
                caption=None,
                reply_markup=keyboard,
            )
            sessions = dict((await state.get_data()).get("collage_sessions", {}))
            sessions[str(sent.message_id)] = {
                "models": list(group),
                "cycle": cycle_id,
                "upload": photo_context.get("upload"),
                "upload_file_id": photo_context.get("upload_file_id"),
                "last_photo_file_id": photo_context.get("last_photo_file_id"),
                "aliases": [str(message.message_id)],
            }
            await state.update_data(models_message_id=sent.message_id, collage_sessions=sessions)
            await _delete_busy_messages(state, message.bot, message.chat.id)

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to send collage message for models %s: %s",
                [model.unique_id for model in group],
                exc,
                extra={"collage_fallback_used": True},
            )
            await _send_batch_as_photos(
                message,
                state,
                group,
                reply_markup=keyboard,
                cycle_id=cycle_id,
                photo_context=photo_context,
            )
            return
        logger.debug(
            "Batch %sx%s delivered for models %s",
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
        cycle_id: int,
        photo_context: dict[str, Any],
    ) -> None:
        last_index = len(group) - 1
        last_sent = None  # ← [ДОБАВЬ ЭТО]

        for index, item in enumerate(group):
            caption = None
            markup = reply_markup if index == last_index else None
            try:
                last_sent = await message.answer_photo(   # ← [ИЗМЕНИ: сохраняем отправленное сообщение]
                    photo=URLInputFile(item.img_user_url),
                    caption=caption,
                    reply_markup=markup,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to send fallback photo for model %s: %s",
                    item.unique_id,
                    exc,
                    extra={"collage_fallback_used": True},
                )

        # ← [ДОБАВЬ БЛОК НИЖЕ] — после цикла записываем id последнего сообщения с кнопками
        if last_sent:
            sessions = dict((await state.get_data()).get("collage_sessions", {}))
            sessions[str(last_sent.message_id)] = {
                "models": list(group),
                "cycle": cycle_id,
                "upload": photo_context.get("upload"),
                "upload_file_id": photo_context.get("upload_file_id"),
                "last_photo_file_id": photo_context.get("last_photo_file_id"),
                "aliases": [str(message.message_id)],
            }
            await state.update_data(models_message_id=last_sent.message_id, collage_sessions=sessions)
        await _delete_busy_messages(state, message.bot, message.chat.id)


    async def _delete_state_message(message: Message, state: FSMContext, key: str) -> None:
        data = await state.get_data()
        message_id = data.get(key)
        if not message_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, message_id)
        except TelegramBadRequest as exc:
            logger.warning(
                "Не удалось удалить сообщение %s (%s): %s",
                key,
                message_id,
                exc,
                extra={"stage": "MESSAGE_CLEANUP"},
            )
        finally:
            if data.get("last_aux_message_id") == message_id:
                await state.update_data(last_aux_message_id=None)
            await state.update_data(**{key: None})

    async def _delete_contact_prompt_message(
        message: Message,
        state: FSMContext,
    ) -> None:
        data = await state.get_data()
        prompt_id = data.get("contact_prompt_message_id")
        if not prompt_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, int(prompt_id))
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            logger.debug(
                "Failed to delete contact prompt %s: %s",
                prompt_id,
                exc,
                extra={"stage": "CONTACT_PROMPT_DELETE"},
            )
        finally:
            await state.update_data(contact_prompt_message_id=None, last_aux_message_id=None)

    promo_video_missing_warned = False

    @router.message(CommandStart())
    async def handle_start(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id
        previous_state = await state.get_state()
        previous_data = await state.get_data()
        contact_was_active = (
            previous_state == ContactRequest.waiting_for_phone.state
            or bool(previous_data.get("contact_request_active"))
        )
        if previous_data:
            await _cleanup_cycle_messages(message, state, data=previous_data)
            await _delete_phone_invalid_message(message, state, data=previous_data)
        await _delete_last_aux_message(message, state)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        # Ensure legacy reply keyboards are hidden for returning users
        await _dismiss_reply_keyboard(message)
        profile_before = await repository.ensure_user(user_id)
        ignored_phone = profile_before.contact_skip_once or contact_was_active
        await state.clear()
        await repository.reset_user_session(user_id)
        current_cycle = await _start_new_cycle(state, user_id)
        profile = await repository.ensure_user(user_id)
        if contact_was_active and not profile_before.contact_skip_once and not profile.contact_never:
            await repository.set_contact_skip_once(user_id, True)
            ignored_phone = True
        contact_record = await repository.get_user_contact(user_id)
        has_contact = bool(contact_record and contact_record.consent)
        remaining = await repository.remaining_tries(user_id)
        contact_never = profile.contact_never
        if (
            remaining < 2
            and ignored_phone
            and not contact_never
            and not has_contact
        ):
            await repository.set_contact_never(user_id, True)
            await repository.set_contact_skip_once(user_id, False)
            contact_never = True
        contact_cooldown = 0
        if not has_contact and not contact_never and remaining >= 2:
            contact_cooldown = 2
        await track_event(str(user_id), "start")
        text = message.text or ""
        if "ref_" in text:
            parts = text.split()
            if parts and parts[0].startswith("/start") and len(parts) > 1:
                ref_part = parts[1]
                if ref_part.startswith("ref_"):
                    ref_id = ref_part.replace("ref_", "")
                    try:
                        await repository.set_referrer(user_id, int(ref_id))
                    except ValueError:
                        pass
        await state.set_state(TryOnStates.START)
        await state.update_data(
            allow_try_button=False,
            contact_request_cooldown=contact_cooldown,
            phone_bad_attempts=0,
            phone_invalid_message_id=None,
            contact_request_active=False,
            contact_prompt_message_id=None,
            upload=None,
            upload_file_id=None,
            last_photo_file_id=None,
            current_models=[],
            last_batch=[],
            presented_model_ids=[],
            selected_model=None,
            is_generating=False,
            suppress_more_button=False,
            reuse_offer_message_id=None,
            reuse_offer_active=False,
            allow_more_button_next=False,
        )
        promo_video_log = {
            "path": str(promo_video_path),
            "width": None,
            "height": None,
            "source": "none",
        }
        start_message_sent = False
        nonlocal promo_video_missing_warned
        if promo_video_enabled:
            if promo_video_path.exists():
                width_override = None
                height_override = None
                source = "none"
                if promo_video_width is not None and promo_video_height is not None:
                    width_override = promo_video_width
                    height_override = promo_video_height
                    source = "env"
                elif (promo_video_width is not None) != (promo_video_height is not None):
                    logger.warning(
                        "Необходимо одновременно задавать PROMO_VIDEO_WIDTH и PROMO_VIDEO_HEIGHT — значения проигнорированы",
                        extra={"stage": "PROMO_VIDEO_CONFIG"},
                    )
                if width_override is None or height_override is None:
                    probed_width, probed_height, probe_meta = probe_video_size(
                        str(promo_video_path)
                    )
                    probe_source = probe_meta.get("source")
                    if probed_width and probed_height:
                        width_override = probed_width
                        height_override = probed_height
                    if probe_source == "cache":
                        source = "cache"
                    elif probe_source in {"ffprobe", "opencv"}:
                        source = "probe"
                    elif probe_source:
                        source = "none"
                promo_video_log.update(
                    width=width_override,
                    height=height_override,
                    source=source,
                )
                send_kwargs: dict[str, Any] = {
                    "video": FSInputFile(promo_video_path),
                    "caption": msg.START_WELCOME,
                    "reply_markup": start_keyboard(),
                    "supports_streaming": True,
                }
                if width_override and height_override:
                    send_kwargs["width"] = width_override
                    send_kwargs["height"] = height_override
                try:
                    await _send_aux_message(
                        message,
                        state,
                        message.answer_video,
                        **send_kwargs,
                    )
                    start_message_sent = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Не удалось отправить промо-видео %s: %s",
                        promo_video_path,
                        exc,
                        extra={"stage": "PROMO_VIDEO_ERROR"},
                    )
            else:
                if not promo_video_missing_warned:
                    logger.warning(
                        "Промо-видео не найдено по пути %s",
                        promo_video_path,
                        extra={"stage": "PROMO_VIDEO_MISSING"},
                    )
                    promo_video_missing_warned = True
        logger.debug(
            "Promo video parameters: %s",
            json.dumps(promo_video_log, ensure_ascii=False),
        )

        if not start_message_sent:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.START_WELCOME,
                reply_markup=start_keyboard(),
            )
        
        info_domain(
            "bot.handlers",
            "👤 Пользователь открыл старт",
            stage="USER_START",
            user_id=message.from_user.id,
        )

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
        await state.update_data(
            gender=gender,
            first_generated_today=True,
            allow_try_button=False,
        )
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await track_event(str(callback.from_user.id), "gender_selected", value=gender)
        await _delete_state_message(callback.message, state, "gender_prompt_message_id")
        await _send_aux_message(
            callback.message,
            state,
            callback.message.answer,
            msg.PHOTO_INSTRUCTION,
        )
        await callback.answer()
        info_domain(
            "bot.handlers",
            f"⚙️ Выбор: пол={gender}",
            stage="FILTER_SELECTED",
            user_id=callback.from_user.id,
        )

    @router.message(
        StateFilter(TryOnStates.AWAITING_PHOTO, TryOnStates.RESULT, TryOnStates.SHOW_RECS),
        ~F.photo, ~F.text.regexp(r"^/")
    )
    async def reject_non_photo(message: Message, state: FSMContext) -> None:
        text = (message.text or "").strip()
        if text:
            if text.startswith("/"):
                return
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.NOT_PHOTO_WARNING,
        )

    @router.message(
        StateFilter(
            TryOnStates.AWAITING_PHOTO,
            TryOnStates.RESULT,
            TryOnStates.SHOW_RECS,
            TryOnStates.GENERATING,
        ),
        F.photo,
    )
    async def accept_photo(message: Message, state: FSMContext) -> None:
        """Accept a new user photo and spin a dedicated try-on cycle for it without cancelling older ones."""
        user_id = message.from_user.id
        data_before = await state.get_data()
        current_state = await state.get_state()
        has_active_flow = any(
            [
                data_before.get("upload"),
                data_before.get("upload_file_id"),
                data_before.get("last_photo_file_id"),
                data_before.get("current_models"),
                data_before.get("selected_model"),
                data_before.get("is_generating"),
                data_before.get("generation_progress_message_id"),
            ]
        )
        wants_new_cycle = current_state in {
            TryOnStates.SHOW_RECS.state,
            TryOnStates.GENERATING.state,
            TryOnStates.RESULT.state,
        } or has_active_flow
        if wants_new_cycle:
            current_cycle = await _start_new_cycle(state, user_id)
            await state.update_data(
                upload=None,
                upload_file_id=None,
                last_photo_file_id=None,
                selected_model=None,
                current_models=[],
                last_batch=[],
                presented_model_ids=[],
                preload_message_id=None,
                generation_progress_message_id=None,
                models_message_id=None,
                suppress_more_button=False,
                reuse_offer_message_id=None,
                reuse_offer_active=False,
                allow_more_button_next=False,
                is_generating=False,
                current_cycle=current_cycle,
            )
        else:
            current_cycle = await _ensure_current_cycle_id(state, user_id)
            await state.update_data(current_cycle=current_cycle, is_generating=False)
        photo = message.photo[-1]
        await state.set_state(TryOnStates.SHOW_RECS)
        path = await save_user_photo(message)
        await state.update_data(
            upload=path,
            upload_file_id=photo.file_id,
            last_photo_file_id=photo.file_id,
        )
        photo_cycle_id = current_cycle
        photo_context = {
            "upload": path,
            "upload_file_id": photo.file_id,
            "last_photo_file_id": photo.file_id,
        }
        await track_event(str(user_id), "photo_uploaded")
        profile = await repository.ensure_daily_reset(user_id)
        if profile.tries_used == 0:
            await state.update_data(first_generated_today=True)
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await track_event(str(user_id), "daily_limit_hit")
            await _send_aux_message(
                message,
                state,
                message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            info_domain(
                "bot.handlers",
                "🔒 Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
            )
            return
        await _delete_idle_nudge_message(state, message.bot, message.chat.id)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        filters = await _ensure_filters(user_id, state)
        await state.update_data(
            presented_model_ids=[],
            current_models=[],
            last_batch=[],
            suppress_more_button=False,
            reuse_offer_message_id=None,
            reuse_offer_active=False,
            allow_more_button_next=False,
        )
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
            exclude_ids=None,
            cycle_id=photo_cycle_id,
            photo_context=photo_context,
        )
        info_domain(
            "bot.handlers",
            "🖼️ Фото получено",
            stage="USER_SENT_PHOTO",
            user_id=user_id,
            remaining=remaining,
        )

    @router.callback_query(StateFilter(TryOnStates.AWAITING_PHOTO), F.data == "send_new_photo")
    async def request_new_photo(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        data = await state.get_data()
        await _cleanup_cycle_messages(callback.message, state, data=data)
        await _delete_last_aux_message(callback.message, state)
        new_cycle = await _start_new_cycle(state, user_id)
        await state.update_data(collage_sessions={}, result_messages={})
        await _prompt_for_next_photo(
            callback.message,
            state,
            msg.PHOTO_INSTRUCTION,
            cycle_id=new_cycle,
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("pick:"))
    async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
        """Launch generation using the model tied to the tapped collage, honoring its original cycle/photo context."""
        parts = callback.data.split(":", 2)
        if len(parts) == 3:
            _, batch_source_key, model_id = parts
        else:  # fallback for legacy format
            batch_source_key = "unknown"
            model_id = callback.data.replace("pick:", "", 1)
        data = await state.get_data()
        result_messages = dict(data.get("result_messages", {}))
        used_message_ids = set(result_messages.keys())
        message_id_str = str(callback.message.message_id)
        pending_collage_ids = set(data.get("pending_collage_ids", []))
        models_msg_id = data.get("models_message_id")
        sessions = dict(data.get("collage_sessions", {}))
        user_id = callback.from_user.id if callback.from_user else None
        info_domain(
            "bot.handlers",
            "🎯 choose_model: старт обработки клика по коллажу",
            stage="CHOOSE_MODEL_START",
            user_id=user_id,
            message_id=callback.message.message_id,
            callback_data=callback.data,
            models_msg_id=models_msg_id,
            has_session_entry=message_id_str in sessions,
            collage_session_keys=list(sessions.keys()),
            used_message_ids=list(used_message_ids),
        )
        if message_id_str in pending_collage_ids or message_id_str in used_message_ids:
            info_domain(
                "bot.handlers",
                "♻️ choose_model: повторный клик по уже использованному коллажу",
                stage="CHOOSE_MODEL_REPEATED_CLICK_IGNORED",
                user_id=user_id,
                message_id=callback.message.message_id,
                reason="pending" if message_id_str in pending_collage_ids else "used",
            )
            await callback.answer()
            return
        session_key = message_id_str
        session_entry = sessions.get(session_key)
        if not session_entry:
            for key, entry in sessions.items():
                aliases = {str(alias) for alias in entry.get("aliases", [])}
                if message_id_str in aliases:
                    session_key = key
                    session_entry = entry
                    break
        if models_msg_id and models_msg_id == callback.message.message_id:
            try:
                await callback.message.bot.delete_message(callback.message.chat.id, models_msg_id)
            except Exception:
                pass
            await state.update_data(models_message_id=None)

        if not session_entry and not (models_msg_id and callback.message.message_id == models_msg_id):
            info_domain(
                "bot.handlers",
                "⚠️ choose_model: модель недоступна для этого коллажа",
                stage="CHOOSE_MODEL_MODEL_UNAVAILABLE",
                user_id=user_id,
                message_id=callback.message.message_id,
                callback_data=callback.data,
                models_msg_id=models_msg_id,
                has_session_entry=bool(session_entry),
            )
            await callback.answer(msg.MODEL_UNAVAILABLE_ALERT, show_alert=True)
            return

        models_data: List[GlassModel] = []
        generation_cycle = None
        photo_context: dict[str, Any] | None = None
        if session_entry:
            models_data = list(session_entry.get("models", []))
            generation_cycle = session_entry.get("cycle")
            photo_context = {
                "upload": session_entry.get("upload"),
                "upload_file_id": session_entry.get("upload_file_id"),
                "last_photo_file_id": session_entry.get("last_photo_file_id"),
            }
        elif models_msg_id and callback.message.message_id == models_msg_id:
            models_data = list(data.get("current_models", []))
            generation_cycle = data.get("current_cycle")
            photo_context = {
                "upload": data.get("upload"),
                "upload_file_id": data.get("upload_file_id"),
                "last_photo_file_id": data.get("last_photo_file_id"),
            }
        else:
            info_domain(
                "bot.handlers",
                "⚠️ choose_model: модель недоступна для этого коллажа",
                stage="CHOOSE_MODEL_MODEL_UNAVAILABLE",
                user_id=user_id,
                message_id=callback.message.message_id,
                callback_data=callback.data,
                models_msg_id=models_msg_id,
                has_session_entry=bool(session_entry),
            )
            await callback.answer(msg.MODEL_UNAVAILABLE_ALERT, show_alert=True)
            return

        selected = next((model for model in models_data if model.unique_id == model_id), None)
        if not selected:
            info_domain(
                "bot.handlers",
                "⚠️ choose_model: модель недоступна для этого коллажа",
                stage="CHOOSE_MODEL_MODEL_UNAVAILABLE",
                user_id=user_id,
                message_id=callback.message.message_id,
                callback_data=callback.data,
                models_msg_id=models_msg_id,
                has_session_entry=bool(session_entry),
            )
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
            await track_event(str(callback.from_user.id), "daily_limit_hit")
            await _send_aux_message(
                callback.message,
                state,
                callback.message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            info_domain(
                "bot.handlers",
                "daily_limit_after_pick",
                stage="DAILY_LIMIT",
                user_id=callback.from_user.id,
                context="model_pick",
            )
            await callback.answer()
            return
        await callback.answer()
        if generation_cycle is None:
            generation_cycle = await _ensure_current_cycle_id(state, callback.from_user.id)
        # Tie generation to the cycle captured for this collage; stale cycles run without locking the current FSM state.
        is_current_cycle = await _is_cycle_current(state, generation_cycle)
        updates: dict[str, Any] = {}
        if is_current_cycle:
            updates["selected_model"] = selected
            if data.get("allow_more_button_next"):
                updates["suppress_more_button"] = False
                updates["allow_more_button_next"] = False
        if updates:
            await state.update_data(**updates)
        if is_current_cycle:
            await state.update_data(is_generating=True)
            await state.set_state(TryOnStates.GENERATING)
        pending_collage_ids.add(message_id_str)
        await state.update_data(pending_collage_ids=list(pending_collage_ids))
        info_domain(
            "bot.handlers",
            "🚀 choose_model: запускаем _perform_generation по коллажу",
            stage="CHOOSE_MODEL_LAUNCH_GENERATION",
            user_id=user_id,
            message_id=callback.message.message_id,
            model_id=model_id,
            generation_cycle=generation_cycle,
            session_key=session_key,
            has_session_entry=bool(session_entry),
        )
        try:
            await _perform_generation(
                callback.message,
                state,
                selected,
                generation_cycle=generation_cycle,
                photo_context=photo_context,
            )
        finally:
            data_after = await state.get_data()
            pending_after = set(data_after.get("pending_collage_ids", []))
            pending_after.discard(message_id_str)
            await state.update_data(pending_collage_ids=list(pending_after))
        sessions.pop(session_key, None)
        await state.update_data(
            collage_sessions=sessions,
            models_message_id=None,
        )
        info_domain(
            "bot.handlers",
            "🧹 choose_model: очистили запись о коллаже из collage_sessions",
            stage="CHOOSE_MODEL_COLLAGE_SESSION_CLEARED",
            user_id=user_id,
            message_id=callback.message.message_id,
            cleared_session_key=session_key,
            remaining_collage_keys=list(sessions.keys()),
        )

    @router.callback_query(
        StateFilter(ContactRequest.waiting_for_phone),
        F.data == CONTACT_SHARE_CALLBACK,
    )
    async def contact_share_button(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if callback.message:
            await _delete_contact_prompt_message(callback.message, state)
        await callback.message.answer(
            msg.ASK_PHONE_PROMPT_MANUAL,  # ← вместо INVISIBLE_PROMPT
            reply_markup=contact_share_reply_keyboard(),
        )


    @router.callback_query(
        StateFilter(ContactRequest.waiting_for_phone),
        F.data == CONTACT_SKIP_CALLBACK,
    )
    async def contact_skip_button(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await _complete_contact_skip(callback.message, state)

    @router.callback_query(
        StateFilter(ContactRequest.waiting_for_phone),
        F.data == CONTACT_NEVER_CALLBACK,
    )
    async def contact_never_button(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        user_id = callback.from_user.id
        if callback.message:
            await _delete_contact_prompt_message(callback.message, state)
            await _dismiss_reply_keyboard(callback.message)
        await _reset_phone_attempts(callback.message, state)
        await repository.set_contact_never(user_id, True)
        await repository.set_contact_skip_once(user_id, False)
        await state.update_data(contact_request_cooldown=0, contact_prompt_due=None, allow_more_button_next=True)
        await _resume_after_contact(callback.message, state, send_generation=False)
        current_state = await state.get_state()
        if current_state != TryOnStates.DAILY_LIMIT_REACHED.state:
            await _send_reuse_prompt(
                callback.message,
                state,
                msg.ASK_PHONE_NEVER_ACK,
            )
        logger.debug("User %s opted out of contacts", user_id)

    @router.message(StateFilter(ContactRequest.waiting_for_phone), F.contact)
    async def contact_shared(message: Message, state: FSMContext) -> None:
        contact = message.contact
        if not contact or not contact.phone_number:
            await _handle_phone_invalid_attempt(message, state)
            return
        normalized = normalize_phone(contact.phone_number)
        if not normalized:
            await _handle_phone_invalid_attempt(message, state)
            return
        await _reset_phone_attempts(message, state)
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
            await _complete_contact_skip(message, state, manual=True)
            return
        if text == msg.ASK_PHONE_BUTTON_NEVER:
            await _reset_phone_attempts(message, state)
            await _dismiss_reply_keyboard(message)
            await repository.set_contact_never(user_id, True)
            await repository.set_contact_skip_once(user_id, False)
            await state.update_data(contact_request_cooldown=0, contact_prompt_due=None)
            await _resume_after_contact(message, state, send_generation=False)
            current_state = await state.get_state()
            if current_state != TryOnStates.DAILY_LIMIT_REACHED.state:
                await _send_reuse_prompt(message, state, msg.ASK_PHONE_NEVER_ACK)
            logger.debug("User %s opted out of contacts", user_id)
            return
        await _handle_manual_phone(message, state, source="manual")

    @router.message(StateFilter(ContactRequest.waiting_for_phone))
    async def contact_fallback(message: Message, state: FSMContext) -> None:
        await _handle_phone_invalid_attempt(message, state)

    async def _perform_generation(
        message: Message,
        state: FSMContext,
        model: GlassModel,
        *,
        generation_cycle: int | None = None,
        photo_context: dict[str, Any] | None = None,
    ) -> None:
        """Run generation and deliver result bound to the provided try-on cycle (stale cycles drop the \"try more\" button)."""

        user_id = message.chat.id
        if generation_cycle is None:
            generation_cycle = await _ensure_current_cycle_id(state, user_id)
        info_domain(
            "bot.handlers",
            "🎬 _perform_generation: старт генерации",
            stage="GEN_START",
            user_id=user_id,
            generation_cycle=generation_cycle,
            message_id=message.message_id,
            model_id=model.unique_id,
        )

        async def _update_if_current(**kwargs: Any) -> None:
            if await _is_cycle_current(state, generation_cycle):
                await state.update_data(**kwargs)

        await _update_if_current(is_generating=True)
        data = await state.get_data()
        upload_value = None
        upload_file_id = None
        last_photo_file_id = None
        if photo_context:
            upload_value = photo_context.get("upload")
            upload_file_id = photo_context.get("upload_file_id")
            last_photo_file_id = photo_context.get("last_photo_file_id")
        if upload_value is None:
            upload_value = data.get("upload")
        if upload_file_id is None:
            upload_file_id = data.get("upload_file_id")
        if last_photo_file_id is None:
            last_photo_file_id = data.get("last_photo_file_id")

        progress_message: Message | None = None
        progress_message_id: int | None = None
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
                    getattr(progress_message, "message_id", None),
                    user_id,
                    exc,
                )
                progress_message = None

        async def _delete_progress_message() -> None:
            nonlocal progress_message, progress_message_id
            removed_id = progress_message_id
            if progress_message_id:
                try:
                    await message.bot.delete_message(message.chat.id, int(progress_message_id))
                except (TelegramBadRequest, TelegramForbiddenError) as exc:
                    logger.debug(
                        "Failed to delete progress message %s: %s",
                        progress_message_id,
                        exc,
                    )
                progress_message_id = None
            progress_message = None
            if await _is_cycle_current(state, generation_cycle):
                data_snapshot = await state.get_data()
                updates: dict[str, Any] = {"generation_progress_message_id": None}
                if removed_id and data_snapshot.get("last_aux_message_id") == removed_id:
                    updates["last_aux_message_id"] = None
                await state.update_data(**updates)

        try:
            if upload_value and Path(upload_value).exists():
                user_photo_path = Path(upload_value)
            elif upload_file_id:
                downloaded = await redownload_user_photo(
                    message.bot, upload_file_id, user_id
                )
                await _update_if_current(upload=downloaded)
                user_photo_path = Path(downloaded)
            elif last_photo_file_id:
                downloaded = await redownload_user_photo(
                    message.bot, last_photo_file_id, user_id
                )
                await _update_if_current(
                    upload=downloaded,
                    upload_file_id=last_photo_file_id,
                )
                user_photo_path = Path(downloaded)
            else:
                raise RuntimeError("User photo is not available")

            progress_message = await _send_aux_message(
                message,
                state,
                message.answer,
                msg.PROGRESS_DOWNLOADING_USER_PHOTO,
            )
            progress_message_id = getattr(progress_message, "message_id", None)
            if progress_message_id:
                await _update_if_current(
                    generation_progress_message_id=int(progress_message_id)
                )

            await asyncio.to_thread(resize_inplace, user_photo_path)
            await _edit_progress(msg.PROGRESS_DOWNLOADING_GLASSES)

            if not model.img_nano_url:
                raise RuntimeError("Frame model does not have NanoBanana reference")
            glasses_path = await fetch_drive_file(model.img_nano_url)
            await asyncio.to_thread(resize_inplace, glasses_path)

            await _edit_progress(msg.PROGRESS_SENDING_TO_GENERATION)
            await _edit_progress(msg.PROGRESS_WAIT_GENERATION)

            await track_event(str(user_id), "generation_started", value=model.unique_id)
            info_domain(
                "generation.nano",
                f"🛠️ Генерация запущена — frame={model.unique_id}",
                stage="GENERATION_STARTED",
                user_id=user_id,
            )

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
            await track_event(str(user_id), "generation_finished", value=str(latency_ms))
            info_domain(
                "generation.nano",
                f"✅ Генерация готова — {latency_ms} ms",
                stage="GENERATION_FINISHED",
                user_id=user_id,
                model_id=model.unique_id,
                latency_ms=latency_ms,
                result_kb=round(result_kb, 1),
                finish_reason=generation_result.finish_reason,
                attempt=generation_result.attempt,
                retried=generation_result.retried,
            )

        except NanoBananaGenerationError as exc:
            latency_ms = (
                int((time.perf_counter() - start_time) * 1000)
                if start_time
                else 0
            )
            logger.error(
                (
                    "NanoBanana не смогла сгенерировать результат: frame=%s finish=%s latency_ms=%s "
                    "inline=%s data_url=%s file_uri=%s detail=%s"
                ),
                model.unique_id,
                exc.finish_reason,
                latency_ms,
                exc.has_inline,
                exc.has_data_url,
                exc.has_file_uri,
                exc.reason_detail,
                extra={
                    "stage": "NANO_ERROR",
                    "payload": {
                        "model_id": model.unique_id,
                        "finish_reason": exc.finish_reason,
                        "reason_detail": exc.reason_detail,
                        "latency_ms": latency_ms,
                    },
                },
            )
            await _delete_progress_message()
            if await _is_cycle_current(state, generation_cycle):
                await _update_if_current(
                    selected_model=None,
                    current_models=[],
                    upload=None,
                    upload_file_id=None,
                    last_photo_file_id=None,
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
                "Сбой генерации: %s (latency_ms=%s)",
                exc,
                latency_ms,
                exc_info=True,
                extra={
                    "stage": "NANO_ERROR",
                    "payload": {"model_id": model.unique_id, "latency_ms": latency_ms},
                },
            )
            await _delete_progress_message()
            if await _is_cycle_current(state, generation_cycle):
                await _update_if_current(
                    selected_model=None,
                    current_models=[],
                    upload=None,
                    upload_file_id=None,
                    last_photo_file_id=None,
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
                    logger.debug(
                        "Не удалось удалить временный файл %s",
                        user_photo_path,
                    )
            if await _is_cycle_current(state, generation_cycle):
                await state.update_data(upload=None, is_generating=False)

        await repository.inc_used_on_success(user_id)
        remaining = await repository.remaining_tries(user_id)
        is_current_cycle = await _is_cycle_current(state, generation_cycle)
        await _delete_progress_message()
        info_domain(
            "bot.handlers",
            "🔍 _perform_generation: статус цикла перед доставкой результата",
            stage="GEN_CYCLE_STATUS",
            user_id=user_id,
            generation_cycle=generation_cycle,
            is_current_cycle=is_current_cycle,
        )
        if not is_current_cycle:
            # Deliver stale cycle in background: keep details button only.
            stale_caption = _compose_result_caption(model, "")
            stale_markup = generation_result_keyboard(
                model.site_url,
                0,
                show_more=False,
            )
            info_domain(
                "bot.handlers",
                "📦 _perform_generation: отправляем stale-результат",
                stage="GEN_DELIVER_STALE",
                user_id=user_id,
                generation_cycle=generation_cycle,
                model_id=model.unique_id,
            )
            await _send_delivery_message(
                message,
                state,
                message.answer_photo,
                BufferedInputFile(result_bytes, filename="result.png"),
                caption=stale_caption,
                reply_markup=stale_markup,
            )
            await repository.increment_generation_count(user_id)
            await repository.register_contact_generation(
                user_id,
                initial_trigger=CONTACT_INITIAL_TRIGGER,
                reminder_trigger=CONTACT_REMINDER_TRIGGER,
            )
            return

        cooldown = max(int(data.get("contact_request_cooldown") or 0), 0)
        if cooldown > 0:
            await state.update_data(contact_request_cooldown=cooldown - 1)
        plan = resolve_generation_followup(
            first_generated_today=data.get("first_generated_today", True),
            remaining=remaining,
        )
        gen_count_before = await repository.get_generation_count(user_id)
        suppress_more = bool(data.get("suppress_more_button"))
        if plan.outcome is GenerationOutcome.FIRST:
            body_text = _render_text(msg.FIRST_RESULT_CAPTION)
        elif plan.outcome is GenerationOutcome.LIMIT:
            body_text = ""
        else:
            followup_index = max(gen_count_before, 1) - 1
            body_text = _resolve_followup_caption(
                followup_index,
                data.get("gender"),
            )
        caption_text = _compose_result_caption(model, body_text)
        result_has_more = plan.remaining > 0
        keyboard_remaining = plan.remaining if result_has_more else 0
        if plan.outcome is GenerationOutcome.LIMIT:
            result_has_more = False
            keyboard_remaining = 0
        if suppress_more:
            result_has_more = False
            keyboard_remaining = 0
        result_markup = generation_result_keyboard(
            model.site_url,
            keyboard_remaining,
            show_more=result_has_more,
        )
        await _deactivate_previous_more_button(message.bot, user_id)
        info_domain(
            "bot.handlers",
            "📦 _perform_generation: отправляем актуальный результат",
            stage="GEN_DELIVER_CURRENT",
            user_id=user_id,
            generation_cycle=generation_cycle,
            model_id=model.unique_id,
        )
        result_message = await _send_delivery_message(
            message,
            state,
            message.answer_photo,
            BufferedInputFile(result_bytes, filename="result.png"),
            caption=caption_text,
            reply_markup=result_markup,
        )
        await _delete_busy_messages(state, message.bot, message.chat.id)
        # === SAVE last_card_message for future trimming (/wear etc.) ===
        try:
            chat_id = result_message.chat.id
        except Exception:
            chat_id = message.chat.id
        await state.update_data(last_card_message={
            "message_id": int(result_message.message_id),
            "chat_id": int(chat_id),
            "type": "caption",
            "title": model.title,
            "trimmed": False,
            "trim_failed": False,
        })
        # === /SAVE last_card_message ===

        info_domain(
            "bot.handlers",
            "📝 _perform_generation: регистрируем результат и alias по source_message_id",
            stage="GEN_REGISTER_RESULT",
            user_id=user_id,
            generation_cycle=generation_cycle,
            source_message_id=message.message_id,
            result_message_id=result_message.message_id,
            has_more=result_has_more,
        )
        await _register_result_message(
            state,
            result_message,
            model,
            has_more=result_has_more,
            source_message_id=message.message_id,
        )
        new_gen_count = await repository.increment_generation_count(user_id)
        daily_gen_count, contact_trigger = await repository.register_contact_generation(
            user_id,
            initial_trigger=CONTACT_INITIAL_TRIGGER,
            reminder_trigger=CONTACT_REMINDER_TRIGGER,
        )
        update_payload = {
            "allow_try_button": True,
            "contact_generations_today": daily_gen_count,
        }
        if contact_trigger:
            update_payload["contact_prompt_due"] = contact_trigger
        await state.update_data(**update_payload)
        new_flag = next_first_flag_value(
            data.get("first_generated_today", True), plan.outcome
        )
        await state.update_data(first_generated_today=new_flag)
        contact_data = await state.get_data()
        contact_active_before = contact_data.get("contact_request_active", False)
        if result_has_more:
            await repository.set_last_more_message(
                user_id,
                result_message.message_id,
                "result",
                {"site_url": model.site_url},
            )
        else:
            await repository.set_last_more_message(user_id, None, None, None)
        if plan.outcome is GenerationOutcome.LIMIT:
            limit_text = _render_text(msg.DAILY_LIMIT_MESSAGE)
            await _delete_last_aux_message(message, state)
            limit_message = await _send_delivery_message(
                message,
                state,
                message.answer,
                limit_text,
                reply_markup=limit_reached_keyboard(site_url),
            )
            await track_event(str(user_id), "daily_limit_hit")
            await state.update_data(last_aux_message_id=limit_message.message_id)
            if contact_active_before:
                await state.update_data(contact_pending_result_state="limit")
            else:
                await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            info_domain(
                "bot.handlers",
                "🔒 Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
                context="post_generation",
            )
        else:
            contact_requested_now = False
            trigger_to_use = update_payload.get("contact_prompt_due")
            if not trigger_to_use:
                trigger_to_use = contact_data.get("contact_prompt_due")
            if not contact_active_before and trigger_to_use:
                contact_requested_now = await _maybe_request_contact(
                    message,
                    state,
                    user_id,
                    origin_state=TryOnStates.RESULT.state,
                    trigger=trigger_to_use,
                )
                if contact_requested_now:
                    logger.debug(
                        "Deferred contact request sent after generation for user %s",
                        user_id,
                    )

            if contact_active_before and not contact_requested_now:
                await state.update_data(contact_pending_result_state="result")
            elif not contact_requested_now:
                await state.set_state(TryOnStates.RESULT)
        logger.debug(
            "Generation result delivered to user %s (model=%s remaining=%s)",
            user_id,
            model.unique_id,
            plan.remaining,
        )

    @router.callback_query(F.data.startswith("more|"))
    async def result_more(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        message = callback.message
        remove_source_message = callback.data in {"more|idle", "more|social"}
        if message is None:
            await callback.answer()
            return
        if await _is_generation_in_progress(state):
            await callback.answer(msg.GENERATION_BUSY)
            return
        current_state = await state.get_state()
        if current_state == ContactRequest.waiting_for_phone.state:
            await callback.answer()
            return
        data_before = await state.get_data()
        if message:
            current_markup = getattr(message, "reply_markup", None)
            updated_markup = remove_more_button(current_markup)
            if updated_markup is not None:
                try:
                    await message.edit_reply_markup(reply_markup=updated_markup)
                except TelegramBadRequest as exc:
                    logger.debug(
                        "more->edit_reply_markup failed for %s: %s",
                        message.message_id, exc
                    )
            data = await state.get_data()
            stored_results = dict(data.get("result_messages", {}))
            entry = stored_results.get(str(message.message_id))
            if entry:
               # сохраняем «Подробнее»: даём в edit_caption ту же клавиатуру, но без «ещё»
                target_markup = updated_markup if updated_markup is not None else current_markup
                try:
                    await message.edit_caption(
                        caption=f"<b>{entry.get('model_title', '')}</b>",
                        reply_markup=target_markup, parse_mode=ParseMode.HTML
                    )
                except TelegramBadRequest as exc:
                    logger.debug(
                       "more->edit_caption failed for %s: %s",
                       message.message_id, exc
                   )
                else:
                    entry["has_more"] = False
                    stored_results[str(message.message_id)] = entry
                    await state.update_data(result_messages=stored_results)
        await repository.set_last_more_message(user_id, None, None, None)
        chat_id = message.chat.id if message else user_id
        await _delete_idle_nudge_message(state, callback.bot, chat_id)
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await track_event(str(user_id), "daily_limit_hit")
            await _send_aux_message(
                callback.message,
                state,
                callback.message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            info_domain(
                "bot.handlers",
                "🔒 Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
                context="more_button",
            )
            await callback.answer()
            return
        upload_exists = bool(data_before.get("upload"))
        upload_file_id = data_before.get("upload_file_id")
        last_photo_file_id = data_before.get("last_photo_file_id")
        active_file_id = upload_file_id or last_photo_file_id
        if not upload_exists and not active_file_id:
            if message:
                await _prompt_for_next_photo(message, state, msg.PHOTO_INSTRUCTION)
            else:
                await state.set_state(TryOnStates.AWAITING_PHOTO)
            if remove_source_message:
                try:
                    await message.bot.delete_message(message.chat.id, message.message_id)
                except TelegramBadRequest as exc:
                    logger.debug(
                        "Failed to delete reminder message %s: %s",
                        message.message_id,
                        exc,
                    )
            await callback.answer()
            return
        filters = await _ensure_filters(user_id, state)
        presented = set(data_before.get("presented_model_ids", []))
        updates: dict[str, Any] = {
            "selected_model": None,
            "current_models": [],
            "last_batch": [],
        }
        if active_file_id:
            updates["upload_file_id"] = active_file_id
        await state.update_data(**updates)
        preload_message = await _send_aux_message(
            message,
            state,
            message.answer,
            msg.SEARCHING_MODELS_PROMPT,
        )
        await state.update_data(preload_message_id=preload_message.message_id)
        await state.set_state(TryOnStates.SHOW_RECS)
        await state.update_data(last_card_message=None)
        success = await _send_models(
            message,
            user_id,
            filters,
            state,
            skip_contact_prompt=True,
            exclude_ids=presented,
        )
        if not success:
            await state.set_state(TryOnStates.RESULT)
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

    @router.callback_query(
        StateFilter(TryOnStates.SHOW_RECS, TryOnStates.RESULT),
        F.data == REUSE_SAME_PHOTO_CALLBACK,
    )
    async def reuse_same_photo(callback: CallbackQuery, state: FSMContext) -> None:
        message = callback.message
        if message is None:
            await callback.answer()
            return
        if await _is_generation_in_progress(state):
            await callback.answer(msg.GENERATION_BUSY)
            return
        user_id = callback.from_user.id
        data = await state.get_data()
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await track_event(str(user_id), "daily_limit_hit")
            await _send_aux_message(
                message,
                state,
                message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            info_domain(
                "bot.handlers",
                "🔒 Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
                context="reuse_photo",
            )
            await callback.answer()
            return
        upload_exists = bool(data.get("upload"))
        upload_file_id = data.get("upload_file_id")
        last_photo_file_id = data.get("last_photo_file_id")
        active_file_id = upload_file_id or last_photo_file_id
        if not upload_exists and not active_file_id:
            await _prompt_for_next_photo(message, state, msg.PHOTO_INSTRUCTION)
            await callback.answer()
            return
        await _delete_idle_nudge_message(state, callback.bot, message.chat.id)
        await _deactivate_previous_more_button(callback.bot, user_id)
        await repository.set_last_more_message(user_id, None, None, None)
        filters = await _ensure_filters(user_id, state)
        presented = set(data.get("presented_model_ids", []))
        reuse_updates: dict[str, Any] = {
            "selected_model": None,
            "current_models": [],
            "last_batch": [],
        }
        if active_file_id:
            reuse_updates["upload_file_id"] = active_file_id
        await state.update_data(**reuse_updates)
        preload_message = await _send_aux_message(
            message,
            state,
            message.answer,
            msg.SEARCHING_MODELS_PROMPT,
        )
        await state.update_data(preload_message_id=preload_message.message_id)
        await state.set_state(TryOnStates.SHOW_RECS)
        success = await _send_models(
            message,
            user_id,
            filters,
            state,
            skip_contact_prompt=True,
            exclude_ids=presented,
        )
        if not success:
            await state.set_state(TryOnStates.RESULT)
        await callback.answer()

    async def start_wear_flow(
        message: Message,
        state: FSMContext,
        *,
        bypass_allow: bool,
        context: str,
    ) -> None:
        current_state = await state.get_state()
        if current_state == ContactRequest.waiting_for_phone.state:
            await message.answer(msg.GENERATION_BUSY)
            return
        if await _is_generation_in_progress(state):
            await message.answer(msg.GENERATION_BUSY)
            return
        data = await state.get_data()
        if not bypass_allow and not data.get("allow_try_button", False):
            return
        user_id = message.from_user.id
        profile = await repository.ensure_user(user_id)
        gender = data.get("gender") or profile.gender
        if gender:
            await state.update_data(gender=gender)
        if not gender:
            await state.set_state(TryOnStates.FOR_WHO)
            prompt_message = await _send_aux_message(
                message,
                state,
                message.answer,
                msg.START_GENDER_PROMPT,
                reply_markup=gender_keyboard(),
            )
            await state.update_data(gender_prompt_message_id=prompt_message.message_id)
            return
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await track_event(str(user_id), "daily_limit_hit")
            await _send_aux_message(
                message,
                state,
                message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            info_domain(
                "bot.handlers",
                "🔒 Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
                context=context,
            )
            return
        prompt_text = _render_text(msg.PHOTO_INSTRUCTION)
        await _prompt_for_next_photo(message, state, prompt_text)

    @router.message(Command("wear"))
    async def command_wear(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id
        data = await state.get_data()
        await _cleanup_cycle_messages(message, state, data=data)
        await _delete_phone_invalid_message(message, state, data=data)
        await _delete_last_aux_message(message, state)
        await _delete_idle_nudge_message(state, message.bot, message.chat.id)
        await _deactivate_previous_more_button(message.bot, user_id)
        await repository.set_last_more_message(user_id, None, None, None)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        profile = await repository.ensure_user(user_id)
        gender = data.get("gender") or profile.gender
        if not gender:
            gender = "male"
            await repository.update_filters(user_id, gender=gender)
        current_cycle = await _start_new_cycle(state, user_id)
        await state.update_data(
            gender=gender,
            current_cycle=current_cycle,
            contact_request_active=False,
            contact_prompt_message_id=None,
            contact_pending_result_state=None,
            contact_prompt_due=None,
            phone_invalid_message_id=None,
            phone_bad_attempts=0,
        )
        remaining = await repository.remaining_tries(user_id)
        if remaining <= 0:
            await state.set_state(TryOnStates.DAILY_LIMIT_REACHED)
            await track_event(str(user_id), "daily_limit_hit")
            await _send_aux_message(
                message,
                state,
                message.answer,
                _render_text(msg.DAILY_LIMIT_MESSAGE),
                reply_markup=limit_reached_keyboard(site_url),
            )
            info_domain(
                "bot.handlers",
                "🔒 Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
                context="wear_command",
            )
            return
        await _delete_idle_nudge_message(state, message.bot, message.chat.id)
        await _deactivate_previous_more_button(message.bot, user_id)
        await repository.set_last_more_message(user_id, None, None, None)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        await state.update_data(
            upload=None,
            upload_file_id=None,
            last_photo_file_id=None,
            selected_model=None,
            current_models=[],
            last_batch=[],
            presented_model_ids=[],
            preload_message_id=None,
            generation_progress_message_id=None,
            contact_prompt_due=None,
            suppress_more_button=False,
            reuse_offer_message_id=None,
            reuse_offer_active=False,
            is_generating=False,
            allow_more_button_next=False,
            result_messages={},
            collage_sessions={},
            models_message_id=None,
        )
        await _trim_last_card_message(message, state, site_url=site_url)
        await state.update_data(last_card_message=None)
        async def _deliver_instruction() -> None:
            if await _edit_last_aux_message(message, state, msg.PHOTO_INSTRUCTION):
                return
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.PHOTO_INSTRUCTION,
            )

        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await _deliver_instruction()

    @router.message(Command("help"))
    async def command_help(message: Message, state: FSMContext) -> None:
        await message.answer(msg.HELP_TEXT, parse_mode=ParseMode.MARKDOWN_V2)

    @router.message(Command("cancel"))
    async def command_cancel(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id
        previous_data = await state.get_data()
        if previous_data:
            await _cleanup_cycle_messages(message, state, data=previous_data)
            await _delete_phone_invalid_message(message, state, data=previous_data)
        await _delete_last_aux_message(message, state)
        await _clear_reuse_offer(state, message.bot, message.chat.id)
        await _deactivate_previous_more_button(message.bot, user_id)
        await repository.set_last_more_message(user_id, None, None, None)
        _cancel_idle_timer(user_id)
        await state.clear()
        await repository.reset_user_session(user_id)
        current_cycle = await _start_new_cycle(state, user_id)
        await state.set_state(TryOnStates.START)
        await state.update_data(
            upload=None,
            upload_file_id=None,
            last_photo_file_id=None,
            selected_model=None,
            current_models=[],
            is_generating=False,
            contact_prompt_due=None,
            suppress_more_button=False,
            reuse_offer_message_id=None,
            reuse_offer_active=False,
            allow_more_button_next=False,
            current_cycle=current_cycle,
        )
        await message.answer(msg.CANCEL_CONFIRMATION)

    @router.message(F.text == msg.MAIN_MENU_TRY_BUTTON)
    async def handle_main_menu_try(message: Message, state: FSMContext) -> None:
        await start_wear_flow(
            message,
            state,
            bypass_allow=False,
            context="try_button",
        )

    @router.message(Command("privacy"))
    async def command_privacy(message: Message, state: FSMContext) -> None:
        markup = privacy_policy_keyboard(policy_button_url)
        if markup:
            await _send_aux_message(
                message,
                state,
                message.answer,
                msg.PRIVACY_POLICY_TEXT,
                reply_markup=markup,
            )
            return
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.PRIVACY_POLICY_TEXT,
        )

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
        logger.debug("Scheduled reminder for user %s", user_id)

    @router.callback_query(F.data == "cta_book")
    async def handle_cta(callback: CallbackQuery) -> None:
        await track_event(str(callback.from_user.id), "cta_book_opened")
        sanitized = (site_url or "").strip()
        if not sanitized:
            await callback.answer(msg.BOOKING_LINK_UNAVAILABLE, show_alert=True)
            return
        follow_markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=sanitized)]]
        )
        await callback.message.answer(msg.BOOKING_OPEN_PROMPT, reply_markup=follow_markup)
        current_markup = getattr(callback.message, "reply_markup", None)
        if current_markup and getattr(current_markup, "inline_keyboard", None):
            new_rows = []
            changed = False
            for row in current_markup.inline_keyboard:
                new_row: list[InlineKeyboardButton] = []
                for button in row:
                    if getattr(button, "callback_data", None) == "cta_book":
                        new_row.append(
                            InlineKeyboardButton(text=msg.BOOKING_BUTTON_TEXT, url=sanitized)
                        )
                        changed = True
                    else:
                        new_row.append(button)
                new_rows.append(new_row)
            if changed:
                replacement = InlineKeyboardMarkup(inline_keyboard=new_rows)
                try:
                    await callback.message.edit_reply_markup(reply_markup=replacement)
                except TelegramBadRequest as exc:
                    logger.debug(
                        "Не удалось обновить клавиатуру после клика по CTA: %s",
                        exc,
                        extra={"stage": "CTA_UPDATE_FAILED"},
                    )
        await callback.answer()

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
