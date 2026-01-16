
"""FSM definitions and handler registration."""

from __future__ import annotations

import asyncio
from email import message
import hashlib
import io
import json
import random
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
    WebAppInfo,
)
from aiogram.types.input_file import BufferedInputFile, FSInputFile, URLInputFile

from app.analytics import track_event
from app.admin.security import is_admin
from app.keyboards import (
    CONTACT_NEVER_CALLBACK,
    CONTACT_SHARE_CALLBACK,
    CONTACT_SKIP_CALLBACK,
    EVENT_BACK_CALLBACK,
    EVENT_FAIL_NEW_PHOTO_CALLBACK,
    EVENT_FAIL_RETRY_CALLBACK,
    EVENT_MORE_CALLBACK,
    EVENT_NEW_PHOTO_CALLBACK,
    EVENT_REUSE_PHOTO_CALLBACK,
    EVENT_TRY_CALLBACK,
    REUSE_SAME_PHOTO_CALLBACK,
    batch_selection_keyboard,
    contact_request_keyboard,
    contact_share_reply_keyboard,
    event_back_to_wear_keyboard,
    event_attempts_exhausted_keyboard,
    event_fail_keyboard,
    event_phone_keyboard,
    event_phone_bonus_keyboard,
    event_trigger_keyboard,
    event_try_more_keyboard,
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
from app.models import FilterOptions, GlassModel, UserContact, UserProfile, STYLE_UNKNOWN
from app.services.catalog_base import CatalogError, CatalogService
from app.config import CollageConfig, DEFAULT_EVENT_MODEL_NAME
from app.media import probe_video_size
from app.services.collage import (
    CollageProcessingError,
    CollageSourceUnavailable,
    build_three_tile_collage,
)
from app.services.contact_export import ContactRecord, ContactSheetExporter
from app.services.event_scenes import EventScenesService
from app.services.event_teaser import (
    EVENT_TEASER_ALREADY_SENT,
    EVENT_TEASER_SENT,
    EVENT_TEASER_SKIPPED,
    EventTeaserResult,
    maybe_send_event_teaser,
)
from app.services.leads_export import LeadPayload, LeadsExporter
from app.services.repository import Repository
from app.infrastructure.concurrency import with_generation_slot
from app.services.drive_fetch import fetch_drive_bytes, fetch_drive_file
from app.services.image_io import (
    redownload_user_photo,
    resize_inplace,
    save_user_photo,
)
from app.services.nanobanana import (
    NanoBananaGenerationError,
    generate_event,
    generate_glasses,
)
from app.recommender import StyleRecommender, VOTE_DUPLICATE, VOTE_INVALID, VOTE_OK
from app.utils.phone import normalize_phone
from app.utils.paths import ensure_dir
from app.texts import messages as msg
from logger import bind_context, get_logger, info_domain, reset_context


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
    catalog: CatalogService,
    style_recommender: StyleRecommender,
    collage_config: CollageConfig,
    collage_builder: Callable[[Sequence[str | None], CollageConfig], Awaitable[io.BytesIO]] = build_three_tile_collage,
    batch_size: int,
    reminder_hours: int,
    selection_button_title_max: int,
    show_model_style_tag: bool,
    site_url: str,
    admin_webapp_url: str | None,
    promo_code: str,
    no_more_message_key: str,
    clear_on_catalog_change: bool,
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
    event_enabled: bool = False,
    event_id: str | None = None,
    event_scenes_sheet: str | None = None,
    event_prompt_json: str | None = None,
    event_debug_bundle: bool = False,
    event_model_name: str | None = None,
) -> Router:
    router = Router()
    # Store admin_webapp_url as router attribute so it can be updated dynamically
    router.admin_webapp_url = admin_webapp_url
    logger = get_logger("bot.handlers")

    idle_delay = max(int(idle_nudge_seconds), 0)
    idle_enabled = enable_idle_nudge and idle_delay > 0
    idle_tasks: dict[int, asyncio.Task] = {}
    collage_locks: dict[int, asyncio.Lock] = {}
    deleted_collage_ids_limit = 200

    def _get_collage_lock(user_id: int) -> asyncio.Lock:
        lock = collage_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            collage_locks[user_id] = lock
        return lock

    policy_url = (privacy_policy_url or "").strip()
    policy_button_url = (
        policy_url or "https://telegra.ph/Politika-konfidencialnosti-LOOV-10-29"
    )
    BUSY_STATES = {TryOnStates.GENERATING.state, TryOnStates.SHOW_RECS.state}
    INVISIBLE_PROMPT = "\u2060"
    EVENT_FREE_LIMIT = 1
    EVENT_PAID_LIMIT = 10
    EVENT_MAX_IN_FLIGHT_PER_USER = 3
    EVENT_FAIL_RETRY_MAX = 2
    EVENT_INFLIGHT_TTL_SEC = 120
    EVENT_NOTICE_WAIT_TTL_SEC = 25
    EVENT_NOTICE_ATTEMPTS_TTL_SEC = 25
    EVENT_NOTICE_LIMIT_TTL_SEC = 15
    EVENT_NOTICE_WAIT_PREVIOUS = "wait_previous"
    EVENT_NOTICE_ATTEMPTS_IN_FLIGHT = "attempts_in_flight"
    EVENT_NOTICE_IN_FLIGHT_LIMIT = "in_flight_limit"
    EVENT_TRIGGER_IMAGE_PATH = (
        Path(__file__).resolve().parent.parent / "images" / "event_new_year.jpg"
    )
    ACTIVE_MODE_WEAR = "WEAR"
    ACTIVE_MODE_EVENT = "EVENT"

    event_key = (event_id or "").strip()
    event_prompt = (event_prompt_json or "").strip()
    event_ready = bool(event_enabled and event_key and event_scenes_sheet and event_prompt)
    event_debug_enabled = bool(event_debug_bundle)
    resolved_event_model_name = (event_model_name or "").strip() or DEFAULT_EVENT_MODEL_NAME
    if resolved_event_model_name.startswith("models/"):
        resolved_event_model_name = resolved_event_model_name.replace("models/", "", 1)
    event_scenes = (
        EventScenesService(event_scenes_sheet) if event_ready else None
    )
    event_inflight: dict[tuple[int, str, int | None], float] = {}
    event_inflight_lock = asyncio.Lock()

    @dataclass(slots=True)
    class EventNoticeRecord:
        message_id: int
        chat_id: int
        token: str

    event_notice_state: dict[tuple[int, str], EventNoticeRecord] = {}
    event_notice_lock = asyncio.Lock()

    if event_enabled and not event_ready:
        logger.warning(
            "Event enabled but not fully configured",
            extra={"stage": "EVENT_CONFIG"},
        )

    def _event_log_human(message: str, *, user_id: int | str | None = None) -> None:
        info_domain("event", message, user_id=user_id, event_human=True)

    def _event_model_label(model_name: str) -> str:
        normalized = model_name.strip()
        if normalized.startswith("models/"):
            normalized = normalized.replace("models/", "", 1)
        if normalized.startswith("gemini-3-pro"):
            return "Gemini 3 Pro"
        return normalized

    event_model_label = _event_model_label(resolved_event_model_name)

    def _event_result_caption(*, attempts_left: int, show_upsell: bool = False) -> str:
        if attempts_left > 0:
            return msg.EVENT_SUCCESS_FOOTER
        if show_upsell:
            return msg.EVENT_SUCCESS_UPSELL
        return msg.EVENT_SUCCESS_FOOTER_EXHAUSTED

    def _event_result_keyboard(
        model: GlassModel | None, *, attempts_left: int
    ) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        site_url = (getattr(model, "site_url", None) or "").strip()
        if site_url:
            base = generation_result_keyboard(
                site_url, remaining=0, show_more=False, vote_payload=None
            )
            rows.extend(base.inline_keyboard)
        if attempts_left > 0:
            rows.extend(event_try_more_keyboard(attempts_left).inline_keyboard)
        else:
            rows.extend(event_back_to_wear_keyboard().inline_keyboard)
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _event_details_markup_from_reply(
        markup: InlineKeyboardMarkup | None,
    ) -> InlineKeyboardMarkup | None:
        if not markup or not markup.inline_keyboard:
            return None
        rows: list[list[InlineKeyboardButton]] = []
        for row in markup.inline_keyboard:
            url_buttons = [button for button in row if getattr(button, "url", None)]
            if url_buttons:
                rows.append(url_buttons)
        if not rows:
            return None
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _event_details_markup_from_url(
        site_url: str | None,
    ) -> InlineKeyboardMarkup | None:
        clean_url = (site_url or "").strip()
        if not clean_url:
            return None
        return generation_result_keyboard(clean_url, remaining=0, show_more=False)

    async def _cleanup_event_result_message(
        bot: Bot,
        *,
        chat_id: int,
        message_id: int,
        reason: str,
        user_id: int | None = None,
        message: Message | None = None,
        site_url: str | None = None,
    ) -> None:
        markup = _event_details_markup_from_reply(
            message.reply_markup if message else None
        )
        if markup is None:
            markup = _event_details_markup_from_url(site_url)
        edit_error: Exception | None = None
        try:
            if message is not None:
                await message.edit_caption(caption="", reply_markup=markup)
            else:
                await bot.edit_message_caption(
                    chat_id=chat_id,
                    message_id=message_id,
                    caption="",
                    reply_markup=markup,
                )
            return
        except AttributeError as exc:
            edit_error = exc
        except TelegramBadRequest as exc:
            if "message is not modified" in str(exc).lower():
                return
            edit_error = exc
        except TelegramForbiddenError as exc:
            edit_error = exc
        try:
            if message is not None:
                await message.edit_reply_markup(reply_markup=markup)
            else:
                await bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=markup,
                )
            return
        except AttributeError as exc:
            edit_error = exc
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            edit_error = exc
        if edit_error:
            logger.debug(
                "Failed to cleanup event result %s: %s",
                message_id,
                edit_error,
                extra={"stage": "EVENT_RESULT_CLEANUP", "reason": reason, "user_id": user_id},
            )

    async def _cleanup_last_event_result(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        reason: str,
        data: Mapping[str, Any] | None = None,
        message_ref: Message | None = None,
    ) -> None:
        payload = dict(data or await state.get_data())
        message_id = payload.get("last_event_result_message_id")
        if not message_id:
            if message_ref:
                await _cleanup_event_result_message(
                    message.bot,
                    chat_id=message.chat.id,
                    message_id=message_ref.message_id,
                    reason=reason,
                    user_id=user_id,
                    message=message_ref,
                )
            return
        site_url = payload.get("last_event_result_site_url")
        try:
            resolved_id = int(message_id)
        except (TypeError, ValueError):
            return
        if not site_url:
            session_entry = dict(payload.get("event_sessions", {})).get(
                str(resolved_id)
            )
            if session_entry:
                site_url = session_entry.get("site_url")
        if message_ref and message_ref.message_id != resolved_id:
            message_ref = None
        await _cleanup_event_result_message(
            message.bot,
            chat_id=message.chat.id,
            message_id=resolved_id,
            reason=reason,
            user_id=user_id,
            message=message_ref,
            site_url=site_url,
        )

    async def _delete_event_aux_message(
        message: Message,
        state: FSMContext,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(data or await state.get_data())
        message_id = payload.get("last_event_aux_message_id")
        if not message_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, int(message_id))
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            logger.debug(
                "Failed to delete event aux message %s: %s",
                message_id,
                exc,
            )
        finally:
            await state.update_data(last_event_aux_message_id=None)

    async def _delete_event_exhausted_message(
        message: Message,
        state: FSMContext,
        *,
        data: Mapping[str, Any] | None = None,
        user_id: int | None = None,
    ) -> None:
        payload = dict(data or await state.get_data())
        message_id = payload.get("last_event_exhausted_message_id")
        if not message_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, int(message_id))
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            info_domain(
                "event",
                "Event postprocess Telegram error",
                stage="EVENT_POSTPROCESS_TG_ERROR",
                user_id=user_id,
                event_id=event_key,
                action="delete_event_exhausted_message",
                error_type=exc.__class__.__name__,
            )
        finally:
            updates: dict[str, Any] = {"last_event_exhausted_message_id": None}
            if payload.get("event_attempts_message_id") == message_id:
                updates["event_attempts_message_id"] = None
            if payload.get("contact_prompt_message_id") == message_id:
                updates["contact_prompt_message_id"] = None
            await state.update_data(**updates)

    def _format_model_button_label(model: GlassModel) -> str:
        title = (model.title or "").strip()
        if not show_model_style_tag:
            return title
        style = (getattr(model, "style", None) or STYLE_UNKNOWN).strip() or STYLE_UNKNOWN
        return f"{title} [{style}]"

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
        vote_payload: Mapping[str, str] | None = None,
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
        if vote_payload:
            entry["vote_payload"] = dict(vote_payload)
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
        vote_payload = entry.get("vote_payload")
        if vote_payload is not None and not isinstance(vote_payload, Mapping):
            vote_payload = None
        current_markup = generation_result_keyboard(
            site_url, remaining=0, vote_payload=vote_payload
        )

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

    def _new_event_rid() -> str:
        return uuid.uuid4().hex[:8]

    def _event_mime_from_path(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        return "image/jpeg"

    def _extension_from_mime(mime: str) -> str:
        normalized = (mime or "").lower()
        if "jpeg" in normalized or "jpg" in normalized:
            return "jpg"
        if "png" in normalized:
            return "png"
        if "webp" in normalized:
            return "webp"
        return "bin"

    def _bytes_signature(data: bytes, length: int = 16) -> str:
        return data[:length].hex()

    def _detect_bytes_kind(data: bytes) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        stripped = data.lstrip()
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            return "application/json"
        if stripped.startswith(b"<"):
            return "text/html"
        return "unknown"

    def _safe_write_debug_bytes(
        path: Path, payload: bytes, *, user_id: int, label: str
    ) -> None:
        try:
            path.write_bytes(payload)
        except OSError as exc:
            info_domain(
                "event",
                "Event debug write failed",
                stage="EVENT_DEBUG_WRITE_FAILED",
                user_id=user_id,
                event_id=event_key,
                label=label,
                error_type=exc.__class__.__name__,
            )

    def _safe_write_debug_text(
        path: Path, payload: str, *, user_id: int, label: str
    ) -> None:
        try:
            path.write_text(payload, encoding="utf-8")
        except OSError as exc:
            info_domain(
                "event",
                "Event debug write failed",
                stage="EVENT_DEBUG_WRITE_FAILED",
                user_id=user_id,
                event_id=event_key,
                label=label,
                error_type=exc.__class__.__name__,
            )

    def _safe_write_debug_json(
        path: Path, payload: Mapping[str, Any], *, user_id: int, label: str
    ) -> None:
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            info_domain(
                "event",
                "Event debug write failed",
                stage="EVENT_DEBUG_WRITE_FAILED",
                user_id=user_id,
                event_id=event_key,
                label=label,
                error_type=exc.__class__.__name__,
            )

    def _normalize_active_mode(value: str | None) -> str:
        if value and value.upper() == ACTIVE_MODE_EVENT:
            return ACTIVE_MODE_EVENT
        return ACTIVE_MODE_WEAR

    def _get_active_mode(data: Mapping[str, Any]) -> str:
        return _normalize_active_mode(data.get("active_mode"))

    async def _set_active_mode(
        state: FSMContext, user_id: int, mode: str, *, source: str
    ) -> None:
        normalized = _normalize_active_mode(mode)
        data = await state.get_data()
        current = _normalize_active_mode(data.get("active_mode"))
        if current == normalized:
            return
        await state.update_data(active_mode=normalized)
        info_domain(
            "event",
            "Mode switched",
            stage="MODE_SWITCH",
            user_id=user_id,
            event_id=event_key,
            source=source,
            from_mode=current,
            to_mode=normalized,
        )

    def _normalize_cycle(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _event_fail_retry_count(data: Mapping[str, Any]) -> int:
        raw = data.get("event_fail_retry_count")
        try:
            value = int(raw or 0)
        except (TypeError, ValueError):
            return 0
        return max(value, 0)

    def _normalize_gender(value: str | None) -> str:
        return (value or "").strip().casefold()

    def _allowed_scene_genders(user_gender: str | None) -> set[str] | None:
        normalized = _normalize_gender(user_gender)
        if normalized in {"male", "for_who_male", "\u043c\u0443\u0436\u0441\u043a\u043e\u0439"}:
            return {"\u043c\u0443\u0436\u0441\u043a\u043e\u0439", "\u0443\u043d\u0438\u0441\u0435\u043a\u0441"}
        if normalized in {"female", "for_who_female", "\u0436\u0435\u043d\u0441\u043a\u0438\u0439"}:
            return {"\u0436\u0435\u043d\u0441\u043a\u0438\u0439", "\u0443\u043d\u0438\u0441\u0435\u043a\u0441"}
        if normalized in {"unisex", "for_who_unisex", "\u0443\u043d\u0438\u0441\u0435\u043a\u0441"}:
            return {"\u0443\u043d\u0438\u0441\u0435\u043a\u0441"}
        return None

    async def _reset_event_fail_retry_count(state: FSMContext) -> None:
        await state.update_data(event_fail_retry_count=0)

    async def _send_event_fail_screen(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        retry_count: int,
    ) -> None:
        await _delete_event_aux_message(message, state)
        allow_retry = retry_count < EVENT_FAIL_RETRY_MAX
        text = msg.EVENT_FAIL_TEXT if allow_retry else msg.EVENT_FAIL_RETRY_LIMIT_TEXT
        sent = await message.answer(
            text,
            reply_markup=event_fail_keyboard(allow_retry=allow_retry),
        )
        await state.update_data(last_event_aux_message_id=sent.message_id)
        info_domain(
            "event",
            "Event fail screen shown",
            stage="EVENT_FAIL_SCREEN_SHOWN",
            user_id=user_id,
            event_id=event_key,
            retry_count=retry_count,
            allow_retry=allow_retry,
        )

    def _extract_event_photo_context(
        data: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        upload = data.get("event_upload")
        upload_file_id = data.get("event_upload_file_id")
        last_photo_file_id = data.get("event_last_photo_file_id")
        if not (upload or upload_file_id or last_photo_file_id):
            return None
        return {
            "upload": upload,
            "upload_file_id": upload_file_id,
            "last_photo_file_id": last_photo_file_id,
        }

    def _extract_wear_photo_context(
        data: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        upload = data.get("upload")
        upload_file_id = data.get("upload_file_id")
        last_photo_file_id = data.get("last_photo_file_id")
        if not (upload or upload_file_id or last_photo_file_id):
            return None
        return {
            "upload": upload,
            "upload_file_id": upload_file_id,
            "last_photo_file_id": last_photo_file_id,
        }

    def _has_photo_context(context: Mapping[str, Any] | None) -> bool:
        if not context:
            return False
        return bool(
            context.get("upload")
            or context.get("upload_file_id")
            or context.get("last_photo_file_id")
        )

    async def _store_event_photo_context(
        state: FSMContext,
        context: Mapping[str, Any],
        *,
        cycle_id: int | None = None,
    ) -> None:
        updates = {
            "event_upload": context.get("upload"),
            "event_upload_file_id": context.get("upload_file_id"),
            "event_last_photo_file_id": context.get("last_photo_file_id"),
        }
        if cycle_id is not None:
            updates["event_current_cycle"] = cycle_id
        await state.update_data(**updates)

    async def _seed_event_photo_from_wear(
        state: FSMContext,
        *,
        user_id: int,
    ) -> tuple[dict[str, Any] | None, int | None]:
        data = await state.get_data()
        if _extract_event_photo_context(data):
            return None, None
        wear_context = _extract_wear_photo_context(data)
        if not wear_context:
            return None, None
        cycle_id = _normalize_cycle(data.get("current_cycle"))
        if cycle_id is None:
            cycle_id = await _ensure_current_cycle_id(state, user_id)
        await _store_event_photo_context(state, wear_context, cycle_id=cycle_id)
        return wear_context, cycle_id

    async def _register_event_session(
        state: FSMContext,
        message: Message,
        *,
        cycle_id: int | None,
        photo_context: Mapping[str, Any],
        site_url: str | None,
    ) -> None:
        resolved_cycle = _normalize_cycle(cycle_id)
        if resolved_cycle is None:
            return
        data = await state.get_data()
        sessions = dict(data.get("event_sessions", {}))
        sessions[str(message.message_id)] = {
            "upload": photo_context.get("upload"),
            "upload_file_id": photo_context.get("upload_file_id"),
            "last_photo_file_id": photo_context.get("last_photo_file_id"),
            "cycle": resolved_cycle,
            "site_url": site_url,
        }
        if len(sessions) > 100:
            keep_keys = list(sessions.keys())[-100:]
            sessions = {key: sessions[key] for key in keep_keys}
        await state.update_data(event_sessions=sessions)

    def _log_event_user_id_sanity(
        *,
        user_id: int,
        bot_id: int,
        chat_id: int | None,
        from_user_id: int | None,
        source: str,
    ) -> bool:
        info_domain(
            "event",
            "Event user_id sanity",
            stage="EVENT_USER_ID_SANITY",
            user_id=user_id,
            event_id=event_key,
            source=source,
            bot_id=bot_id,
            chat_id=chat_id,
            from_user_id=from_user_id,
        )
        if user_id == bot_id:
            info_domain(
                "event",
                "Event user_id bug",
                stage="EVENT_USER_ID_BUG",
                user_id=user_id,
                event_id=event_key,
                source=source,
                bot_id=bot_id,
                chat_id=chat_id,
                from_user_id=from_user_id,
            )
            return False
        return True

    async def _safe_answer_callback(
        callback: CallbackQuery, *, user_id: int, source: str
    ) -> None:
        started_at = time.perf_counter()
        try:
            await callback.answer()
            delta_ms = int((time.perf_counter() - started_at) * 1000)
            info_domain(
                "event",
                "Event callback answered",
                stage="EVENT_CALLBACK_ANSWERED",
                user_id=user_id,
                event_id=event_key,
                source=source,
                delta_ms_from_received=delta_ms,
            )
        except Exception as exc:  # noqa: BLE001
            delta_ms = int((time.perf_counter() - started_at) * 1000)
            info_domain(
                "event",
                "Event callback answer failed",
                stage="EVENT_CALLBACK_ANSWER_FAILED",
                user_id=user_id,
                event_id=event_key,
                source=source,
                delta_ms_from_received=delta_ms,
                error_type=exc.__class__.__name__,
                error_detail=str(exc),
            )

    async def _maybe_delete_event_attempts_message(
        message: Message, state: FSMContext, user_id: int
    ) -> None:
        data = await state.get_data()
        message_id = data.get("event_attempts_message_id")
        if not message_id:
            return
        if message.message_id != message_id:
            return
        try:
            await message.bot.delete_message(message.chat.id, int(message_id))
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            info_domain(
                "event",
                "Event postprocess Telegram error",
                stage="EVENT_POSTPROCESS_TG_ERROR",
                user_id=user_id,
                event_id=event_key,
                action="delete_event_attempts_message",
                error_type=exc.__class__.__name__,
            )
        finally:
            updates: dict[str, Any] = {"event_attempts_message_id": None}
            if data.get("contact_prompt_message_id") == message_id:
                updates["contact_prompt_message_id"] = None
            if data.get("last_event_exhausted_message_id") == message_id:
                updates["last_event_exhausted_message_id"] = None
            await state.update_data(**updates)

    async def _acquire_event_lock(user_id: int, *, cycle_id: int | None) -> bool:
        key = (user_id, event_key, cycle_id)
        now = time.time()
        async with event_inflight_lock:
            stale = [
                item_key
                for item_key, ts in event_inflight.items()
                if now - ts > EVENT_INFLIGHT_TTL_SEC
            ]
            for item_key in stale:
                event_inflight.pop(item_key, None)
            if key in event_inflight:
                return False
            event_inflight[key] = now
            return True

    async def _release_event_lock(user_id: int, *, cycle_id: int | None) -> None:
        key = (user_id, event_key, cycle_id)
        async with event_inflight_lock:
            event_inflight.pop(key, None)

    async def _event_phone_present(user_id: int) -> bool:
        contact = await repository.get_user_contact(user_id)
        return bool(contact and contact.consent)

    async def _event_attempts_snapshot(
        user_id: int, phone_present: bool
    ) -> tuple[bool, int, int, bool, int, int]:
        free_unlocked, free_used, paid_used = await repository.get_event_attempts(
            user_id, event_key
        )
        free_available = free_unlocked and free_used < EVENT_FREE_LIMIT
        paid_remaining = (
            max(EVENT_PAID_LIMIT - paid_used, 0) if phone_present else 0
        )
        attempts_left = (1 if free_available else 0) + paid_remaining
        return (
            free_unlocked,
            free_used,
            paid_used,
            free_available,
            paid_remaining,
            attempts_left,
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

    def _event_attempts_left(
        *, phone_present: bool, free_unlocked: bool, free_used: int, paid_used: int
    ) -> int:
        free_available = free_unlocked and free_used < EVENT_FREE_LIMIT
        paid_remaining = max(EVENT_PAID_LIMIT - paid_used, 0) if phone_present else 0
        return (1 if free_available else 0) + paid_remaining

    async def _delete_event_notice(
        bot: Bot,
        *,
        user_id: int,
        notice_type: str,
        token: str | None = None,
    ) -> bool:
        key = (user_id, notice_type)
        async with event_notice_lock:
            record = event_notice_state.get(key)
            if not record:
                return False
            if token is None:
                token = record.token
            if record.token != token:
                return False
            chat_id = record.chat_id
            message_id = record.message_id
        try:
            await bot.delete_message(chat_id, int(message_id))
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            logger.debug(
                "Failed to delete event notice %s for %s: %s",
                notice_type,
                user_id,
                exc,
            )
        finally:
            async with event_notice_lock:
                current = event_notice_state.get(key)
                if current and current.token == token:
                    event_notice_state.pop(key, None)
        return True

    async def _expire_event_notice(
        bot: Bot,
        *,
        user_id: int,
        notice_type: str,
        token: str,
        ttl_seconds: int,
    ) -> None:
        try:
            await asyncio.sleep(ttl_seconds)
            await _delete_event_notice(
                bot, user_id=user_id, notice_type=notice_type, token=token
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Event notice TTL cleanup failed for %s %s: %s",
                user_id,
                notice_type,
                exc,
            )

    async def _send_event_notice(
        message: Message,
        *,
        user_id: int,
        notice_type: str,
        text: str,
        ttl_seconds: int,
    ) -> None:
        token = uuid.uuid4().hex
        key = (user_id, notice_type)
        async with event_notice_lock:
            record = event_notice_state.get(key)
            if record:
                record.token = token
                asyncio.create_task(
                    _expire_event_notice(
                        message.bot,
                        user_id=user_id,
                        notice_type=notice_type,
                        token=token,
                        ttl_seconds=ttl_seconds,
                    )
                )
                return
            try:
                sent = await message.answer(text)
            except (TelegramBadRequest, TelegramForbiddenError) as exc:
                logger.debug(
                    "Failed to send event notice %s for %s: %s",
                    notice_type,
                    user_id,
                    exc,
                )
                return
            event_notice_state[key] = EventNoticeRecord(
                message_id=sent.message_id,
                chat_id=sent.chat.id,
                token=token,
            )
            asyncio.create_task(
                _expire_event_notice(
                    message.bot,
                    user_id=user_id,
                    notice_type=notice_type,
                    token=token,
                    ttl_seconds=ttl_seconds,
                )
            )

    async def _event_attempts_gate(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        phone_present: bool,
        free_unlocked: bool,
        attempts_left: int,
        reserved: int,
        max_in_flight: int,
    ) -> bool:
        if attempts_left <= 0:
            if reserved > 0:
                if phone_present:
                    await _send_event_notice(
                        message,
                        user_id=user_id,
                        notice_type=EVENT_NOTICE_ATTEMPTS_IN_FLIGHT,
                        text=msg.EVENT_ATTEMPTS_IN_FLIGHT,
                        ttl_seconds=EVENT_NOTICE_ATTEMPTS_TTL_SEC,
                    )
                else:
                    await _send_event_notice(
                        message,
                        user_id=user_id,
                        notice_type=EVENT_NOTICE_WAIT_PREVIOUS,
                        text=msg.EVENT_WAIT_PREVIOUS,
                        ttl_seconds=EVENT_NOTICE_WAIT_TTL_SEC,
                    )
                return False
            if not phone_present and not free_unlocked:
                await message.answer(msg.EVENT_NO_ACCESS)
                return False
            await _send_event_attempts_exhausted(
                message,
                state,
                user_id=user_id,
                phone_present=phone_present,
            )
            return False
        if max_in_flight > 0 and reserved >= max_in_flight:
            if phone_present:
                await _send_event_notice(
                    message,
                    user_id=user_id,
                    notice_type=EVENT_NOTICE_IN_FLIGHT_LIMIT,
                    text=msg.EVENT_IN_FLIGHT_LIMIT,
                    ttl_seconds=EVENT_NOTICE_LIMIT_TTL_SEC,
                )
            else:
                await _send_event_notice(
                    message,
                    user_id=user_id,
                    notice_type=EVENT_NOTICE_WAIT_PREVIOUS,
                    text=msg.EVENT_WAIT_PREVIOUS,
                    ttl_seconds=EVENT_NOTICE_WAIT_TTL_SEC,
                )
            return False
        return True

    async def _event_preflight_attempt(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        phone_present: bool,
    ) -> bool:
        free_unlocked, free_used, paid_used, reserved = (
            await repository.get_event_attempts_state(user_id, event_key)
        )
        attempts_available = _event_attempts_left(
            phone_present=phone_present,
            free_unlocked=free_unlocked,
            free_used=free_used,
            paid_used=paid_used,
        )
        attempts_left = attempts_available - reserved
        max_in_flight = EVENT_MAX_IN_FLIGHT_PER_USER if phone_present else 1
        return await _event_attempts_gate(
            message,
            state,
            user_id=user_id,
            phone_present=phone_present,
            free_unlocked=free_unlocked,
            attempts_left=attempts_left,
            reserved=reserved,
            max_in_flight=max_in_flight,
        )

    async def _cleanup_event_notices(
        bot: Bot,
        *,
        user_id: int,
        max_in_flight: int,
    ) -> None:
        if not event_key:
            return
        try:
            _, _, _, reserved = await repository.get_event_attempts_state(
                user_id, event_key
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to read event attempts for notice cleanup %s: %s",
                user_id,
                exc,
            )
            return
        if reserved <= 0:
            await _delete_event_notice(
                bot, user_id=user_id, notice_type=EVENT_NOTICE_WAIT_PREVIOUS
            )
        await _delete_event_notice(
            bot, user_id=user_id, notice_type=EVENT_NOTICE_ATTEMPTS_IN_FLIGHT
        )
        if max_in_flight > 0 and reserved < max_in_flight:
            await _delete_event_notice(
                bot, user_id=user_id, notice_type=EVENT_NOTICE_IN_FLIGHT_LIMIT
            )

    async def _handle_wear_success_event(
        message: Message,
        *,
        user_id: int,
        generation_cycle: int,
        generation_id: str | None,
        source_message_id: int | None,
        result_message_id: int | None,
        photo_file_id: str | None,
        model_id: str | None,
        source: str,
    ) -> None:
        if not event_key:
            return
        result = await _maybe_send_event_teaser(
            message,
            user_id,
            generation_cycle=generation_cycle,
            generation_id=generation_id,
            source_message_id=source_message_id,
            result_message_id=result_message_id,
            photo_file_id=photo_file_id,
            model_id=model_id,
            source=source,
            send_teaser=event_ready,
        )
        if result.unlocked:
            info_domain(
                "event",
                "Event access unlocked after wear success",
                stage="EVENT_ACCESS_UNLOCKED",
                user_id=user_id,
                event_id=event_key,
                generation_cycle=generation_cycle,
                generation_id=generation_id,
                source_message_id=source_message_id,
                result_message_id=result_message_id,
                photo_file_id=photo_file_id,
                model_id=model_id,
                source=source,
            )
            info_domain(
                "event",
                "Wear first success reached",
                stage="WEAR_FIRST_SUCCESS",
                user_id=user_id,
                event_id=event_key,
                generation_cycle=generation_cycle,
                generation_id=generation_id,
                source_message_id=source_message_id,
                result_message_id=result_message_id,
                photo_file_id=photo_file_id,
                model_id=model_id,
                source=source,
            )

    async def _maybe_send_event_teaser(
        message: Message,
        user_id: int,
        *,
        generation_cycle: int | None = None,
        generation_id: str | None = None,
        source_message_id: int | None = None,
        result_message_id: int | None = None,
        photo_file_id: str | None = None,
        model_id: str | None = None,
        source: str | None = None,
        send_teaser: bool = True,
    ) -> EventTeaserResult:
        if not event_key:
            return EventTeaserResult(
                status=EVENT_TEASER_SKIPPED,
                claimed=False,
                unlocked=False,
            )
        payload = {
            "generation_cycle": generation_cycle,
            "generation_id": generation_id,
            "source_message_id": source_message_id,
            "result_message_id": result_message_id,
            "photo_file_id": photo_file_id,
            "model_id": model_id,
            "source": source,
        }
        result = await maybe_send_event_teaser(
            bot=message.bot,
            repository=repository,
            user_id=user_id,
            chat_id=message.chat.id,
            event_id=event_key,
            text=msg.EVENT_TRIGGER_TEXT,
            reply_markup=event_trigger_keyboard(),
            image_path=EVENT_TRIGGER_IMAGE_PATH,
            send_teaser=send_teaser,
            logger=logger,
        )
        if result.status == EVENT_TEASER_SKIPPED:
            return result
        if result.status == EVENT_TEASER_ALREADY_SENT:
            info_domain(
                "event",
                "Event trigger skipped",
                stage="EVENT_TRIGGER_SKIPPED",
                user_id=user_id,
                event_id=event_key,
                reason="already_shown",
                **payload,
            )
            return result
        info_domain(
            "event",
            "Event trigger attempt",
            stage="EVENT_TRIGGER_ATTEMPT",
            user_id=user_id,
            event_id=event_key,
            **payload,
        )
        if result.status == EVENT_TEASER_SENT:
            info_domain(
                "event",
                "Event trigger shown",
                stage="EVENT_TRIGGER_SHOWN",
                user_id=user_id,
                event_id=event_key,
                **payload,
            )
            return result
        info_domain(
            "event",
            "Event trigger failed",
            stage="EVENT_TRIGGER_FAILED",
            user_id=user_id,
            event_id=event_key,
            error_type=result.error_type or "unknown",
            retry_after=result.retry_after,
            **payload,
        )
        return result

    async def _send_event_phone_prompt(
        message: Message, state: FSMContext, user_id: int
    ) -> None:
        data = await state.get_data()
        if data.get("contact_request_active"):
            return
        contact = await repository.get_user_contact(user_id)
        if contact and contact.consent:
            return
        await _dismiss_reply_keyboard(message)
        prompt_message = await message.answer(
            msg.EVENT_PHONE_PROMPT,
            reply_markup=event_phone_keyboard(),
        )
        await state.update_data(
            contact_request_active=True,
            contact_pending_generation=False,
            contact_pending_result_state=None,
            contact_prompt_message_id=prompt_message.message_id,
            contact_prompt_due=None,
            phone_bad_attempts=0,
            phone_invalid_message_id=None,
        )
        await state.set_state(ContactRequest.waiting_for_phone)

    async def _send_event_attempts_exhausted(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        phone_present: bool,
    ) -> None:
        reply_markup = event_attempts_exhausted_keyboard(
            show_phone_button=not phone_present
        )
        exhausted_message = await message.answer(
            msg.EVENT_ATTEMPTS_EXHAUSTED,
            reply_markup=reply_markup,
        )
        await state.update_data(
            event_attempts_message_id=exhausted_message.message_id,
            last_event_exhausted_message_id=exhausted_message.message_id,
        )
        if phone_present:
            return
        await state.update_data(
            contact_request_active=True,
            contact_pending_generation=False,
            contact_pending_result_state=None,
            contact_prompt_message_id=exhausted_message.message_id,
            contact_prompt_due=None,
            phone_bad_attempts=0,
            phone_invalid_message_id=None,
        )
        await state.set_state(ContactRequest.waiting_for_phone)

    async def _prompt_event_photo(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        source: str,
        text: str | None = None,
    ) -> None:
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source=source)
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await message.answer(
            text or msg.EVENT_NEED_PHOTO,
            reply_markup=event_back_to_wear_keyboard(),
        )

    async def _run_event_generation(
        message: Message,
        state: FSMContext,
        *,
        user_id: int,
        source: str,
        attempts_snapshot: tuple[bool, int, int, bool, int, int] | None = None,
        phone_present: bool | None = None,
        photo_context: dict[str, Any] | None = None,
        event_cycle_id: int | None = None,
        charge_attempts: bool = True,
        photo_prompt_text: str | None = None,
    ) -> None:
        if not event_enabled:
            await message.answer(msg.EVENT_DISABLED)
            return
        if not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        retry_count = _event_fail_retry_count(data)
        if await _is_generation_in_progress(state):
            if _get_active_mode(data) == ACTIVE_MODE_WEAR:
                await message.answer(msg.GENERATION_BUSY)
                return
        current_state = await state.get_state()
        if current_state == ContactRequest.waiting_for_phone.state:
            await message.answer(msg.GENERATION_BUSY)
            return

        if attempts_snapshot is None:
            phone_present = await _event_phone_present(user_id)
            (
                free_unlocked,
                free_used,
                paid_used,
                free_available,
                paid_remaining,
                attempts_left,
            ) = await _event_attempts_snapshot(user_id, phone_present)
        else:
            if phone_present is None:
                phone_present = await _event_phone_present(user_id)
            (
                free_unlocked,
                free_used,
                paid_used,
                free_available,
                paid_remaining,
                attempts_left,
            ) = attempts_snapshot
        generation_latency_ms = 0
        max_in_flight = EVENT_MAX_IN_FLIGHT_PER_USER if phone_present else 1

        def _event_log_failure(reason: str, *, attempts_charged: bool = False) -> None:
            charged_label = "да" if attempts_charged else "нет"
            _event_log_human(
                f"❌ Ошибка генерации: {reason}. Попытка списана: {charged_label}",
                user_id=user_id,
            )
        info_domain(
            "event",
            "Event phone present",
            stage="EVENT_PHONE_PRESENT",
            user_id=user_id,
            event_id=event_key,
            phone_present=phone_present,
        )
        info_domain(
            "event",
            "Event entry",
            stage="EVENT_ENTER",
            user_id=user_id,
            event_id=event_key,
            source=source,
            phone_present=phone_present,
            free_available=free_available,
            paid_available=paid_remaining,
        )
        resolved_context = (
            dict(photo_context) if photo_context else _extract_event_photo_context(data)
        )
        if not _has_photo_context(resolved_context):
            await _prompt_event_photo(
                message,
                state,
                user_id=user_id,
                source=source,
                text=photo_prompt_text,
            )
            return
        resolved_cycle = event_cycle_id
        if resolved_cycle is None:
            resolved_cycle = _normalize_cycle(data.get("event_current_cycle"))
        if resolved_cycle is None:
            resolved_cycle = await _ensure_current_cycle_id(state, user_id)
        await _store_event_photo_context(
            state,
            resolved_context or {},
            cycle_id=resolved_cycle,
        )

        if not await _acquire_event_lock(user_id, cycle_id=resolved_cycle):
            info_domain(
                "event",
                "Event inflight rejected",
                stage="EVENT_INFLIGHT_REJECTED",
                user_id=user_id,
                event_id=event_key,
                source=source,
                cycle_id=resolved_cycle,
            )
            await message.answer(msg.GENERATION_BUSY)
            return

        lock_cycle_id = resolved_cycle
        rid = _new_event_rid()
        tokens = bind_context(request_id=rid, user_id=user_id)
        debug_dir: Path | None = None
        debug_meta: dict[str, Any] | None = None
        request_meta_path: Path | None = None
        if event_debug_enabled:
            debug_dir = ensure_dir(Path("./results/debug"))
            debug_meta = {
                "rid": rid,
                "user_id": user_id,
                "event_id": event_key,
                "source": source,
            }
            request_meta_path = debug_dir / f"request_meta_{rid}.json"
        progress_message: Message | None = None
        progress_message_id: int | None = None
        user_photo_path: Path | None = None
        start_time = 0.0
        source_kind: str | None = None
        reservation = None
        attempt_finalized = False

        def _write_debug_meta() -> None:
            if not debug_dir or not debug_meta or not request_meta_path:
                return
            _safe_write_debug_json(
                request_meta_path,
                debug_meta,
                user_id=user_id,
                label="request_meta",
            )

        async def _edit_progress(text: str) -> None:
            nonlocal progress_message
            if not progress_message:
                return
            try:
                await progress_message.edit_text(text)
            except TelegramBadRequest as exc:
                logger.debug(
                    "Failed to edit event progress message %s: %s",
                    getattr(progress_message, "message_id", None),
                    exc,
                )
                progress_message = None

        async def _delete_progress() -> None:
            nonlocal progress_message, progress_message_id
            if progress_message_id:
                try:
                    await message.bot.delete_message(
                        message.chat.id, int(progress_message_id)
                    )
                except (TelegramBadRequest, TelegramForbiddenError) as exc:
                    info_domain(
                        "event",
                        "Event postprocess Telegram error",
                        stage="EVENT_POSTPROCESS_TG_ERROR",
                        user_id=user_id,
                        event_id=event_key,
                        action="delete_progress_message",
                        error_type=exc.__class__.__name__,
                    )
            progress_message = None
            progress_message_id = None

        try:
            reservation = await repository.reserve_event_attempt(
                user_id,
                event_key,
                rid,
                phone_present=phone_present,
                max_in_flight=max_in_flight,
            )
            info_domain(
                "event",
                "Event attempt reserve",
                stage="EVENT_ATTEMPT_RESERVE",
                user_id=user_id,
                event_id=event_key,
                rid=rid,
                ok=reservation.ok,
                reason=reservation.reason,
                use_free=reservation.use_free,
                reserved=reservation.reserved,
                attempts_available=reservation.attempts_available,
                attempts_left=reservation.attempts_left,
                max_in_flight=max_in_flight,
                phone_present=phone_present,
                free_unlocked=reservation.free_unlocked,
                free_used=reservation.free_used,
                paid_used=reservation.paid_used,
            )
            if not reservation.ok:
                if reservation.reason in {"no_attempts", "in_flight_limit"}:
                    await _event_attempts_gate(
                        message,
                        state,
                        user_id=user_id,
                        phone_present=phone_present,
                        free_unlocked=reservation.free_unlocked,
                        attempts_left=reservation.attempts_left,
                        reserved=reservation.reserved,
                        max_in_flight=max_in_flight,
                    )
                    return
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return

            attempts_left_before = reservation.attempts_left
            if reservation.reason is None:
                attempts_left_before += 1

            _event_log_human(
                f"📸 Пользователь отправил фото (event_id={event_key}, попыток до: {attempts_left_before})",
                user_id=user_id,
            )
            _event_log_human(
                f"🧠 Модель генерации: {event_model_label} (EVENT)",
                user_id=user_id,
            )
            info_domain(
                "event",
                "Event try request",
                stage="EVENT_TRY_REQUEST",
                user_id=user_id,
                event_id=event_key,
                attempts_left_before=attempts_left_before,
                rid=rid,
            )

            progress_message = await message.answer(msg.EVENT_STARTING)
            progress_message_id = getattr(progress_message, "message_id", None)

            upload_value = None
            upload_file_id = None
            last_photo_file_id = None
            if resolved_context:
                upload_value = resolved_context.get("upload")
                upload_file_id = resolved_context.get("upload_file_id")
                last_photo_file_id = resolved_context.get("last_photo_file_id")
            file_id = upload_file_id or last_photo_file_id
            face_source = "event_upload"
            source_kind = "path"
            if upload_value and Path(upload_value).exists():
                user_photo_path = Path(upload_value)
            elif file_id:
                face_source = (
                    "event_upload_file_id"
                    if upload_file_id
                    else "event_last_photo_file_id"
                )
                source_kind = "file_id"
                downloaded = await redownload_user_photo(
                    message.bot, file_id, user_id
                )
                user_photo_path = Path(downloaded)
            else:
                await _prompt_event_photo(
                    message,
                    state,
                    user_id=user_id,
                    source=source,
                    text=photo_prompt_text,
                )
                info_domain(
                    "event",
                    "Event face missing",
                    stage="EVENT_FACE_MISSING",
                    user_id=user_id,
                    event_id=event_key,
                    source=face_source,
                )
                _event_log_failure("не удалось получить фото")
                return
            await asyncio.to_thread(resize_inplace, user_photo_path)
            face_bytes = user_photo_path.read_bytes()
            face_mime = _event_mime_from_path(user_photo_path)
            if not face_bytes:
                await _prompt_event_photo(
                    message,
                    state,
                    user_id=user_id,
                    source=source,
                    text=photo_prompt_text,
                )
                info_domain(
                    "event",
                    "Event face missing",
                    stage="EVENT_FACE_MISSING",
                    user_id=user_id,
                    event_id=event_key,
                    source=face_source,
                )
                _event_log_failure("не удалось получить фото")
                return
            face_hash = hashlib.sha256(face_bytes).hexdigest()[:12]
            if debug_dir and debug_meta:
                face_ext = _extension_from_mime(face_mime)
                _safe_write_debug_bytes(
                    debug_dir / f"face_{rid}.{face_ext}",
                    face_bytes,
                    user_id=user_id,
                    label="face",
                )
                debug_meta.update(
                    {
                        "face": {
                            "bytes": len(face_bytes),
                            "hash": face_hash,
                            "mime": face_mime,
                            "source": face_source,
                            "source_kind": source_kind,
                        }
                    }
                )
                _write_debug_meta()
            info_domain(
                "event",
                "Face selected",
                stage="EVENT_SELECT_FACE",
                user_id=user_id,
                event_id=event_key,
                source=face_source,
                source_kind=source_kind,
                face_bytes=len(face_bytes),
                face_hash=face_hash,
                face_mime=face_mime,
            )

            await _edit_progress(msg.PROGRESS_DOWNLOADING_GLASSES)

            try:
                snapshot = await catalog.snapshot()
            except CatalogError as exc:
                info_domain(
                    "event",
                    "Event failed: catalog unavailable",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="frame_catalog",
                    error_type=exc.__class__.__name__,
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("не удалось получить каталог оправ")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return
            models = [model for model in snapshot.models if model.img_nano_url]
            if not models:
                info_domain(
                    "event",
                    "Event failed: no frames",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="frame_select",
                    error_type="no_frames",
                    retryable=False,
                    attempts_not_charged=True,
                )
                _event_log_failure("нет доступных оправ")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return
            rng = random.Random()
            rng.shuffle(models)

            frame_model: GlassModel | None = None
            frame_download = None
            frame_attempts = 0
            for idx, model in enumerate(models[: min(len(models), 5)], start=1):
                frame_attempts = idx
                try:
                    download = await fetch_drive_bytes(model.img_nano_url, retries=3)
                except Exception as exc:  # noqa: BLE001
                    info_domain(
                        "event",
                        "Frame download failed",
                        stage="EVENT_SELECT_FRAME",
                        user_id=user_id,
                        event_id=event_key,
                        model_id=model.unique_id,
                        model_code=model.model_code,
                        url=model.img_nano_url,
                        retry_count=idx - 1,
                        ok=False,
                        error_type=exc.__class__.__name__,
                    )
                    continue
                frame_model = model
                frame_download = download
                info_domain(
                    "event",
                    "Frame selected",
                    stage="EVENT_SELECT_FRAME",
                    user_id=user_id,
                    event_id=event_key,
                    model_id=model.unique_id,
                    model_code=model.model_code,
                    url=model.img_nano_url,
                    retry_count=idx - 1,
                    ok=True,
                )
                break
            if frame_model is None or frame_download is None:
                info_domain(
                    "event",
                    "Event failed: frame download",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="frame_select",
                    error_type="frame_download_failed",
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("не удалось скачать оправу")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return
            frame_hash = hashlib.sha256(frame_download.data).hexdigest()[:12]
            if debug_dir and debug_meta:
                frame_ext = frame_download.extension or _extension_from_mime(
                    frame_download.mime
                )
                _safe_write_debug_bytes(
                    debug_dir / f"frame_{rid}.{frame_ext}",
                    frame_download.data,
                    user_id=user_id,
                    label="frame",
                )
                debug_meta.update(
                    {
                        "frame": {
                            "bytes": frame_download.size,
                            "hash": frame_hash,
                            "mime": frame_download.mime,
                            "drive_id": frame_download.drive_id,
                        },
                        "model_id": frame_model.unique_id,
                        "model_code": frame_model.model_code,
                    }
                )
                _write_debug_meta()

            await _edit_progress(msg.EVENT_PROGRESS_DOWNLOADING_SCENE)

            if event_scenes is None:
                info_domain(
                    "event",
                    "Event failed: scenes not configured",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="scene_select",
                    error_type="no_scenes",
                    retryable=False,
                    attempts_not_charged=True,
                )
                _event_log_failure("нет доступных сцен")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return
            try:
                scenes = await event_scenes.list_scenes()
            except Exception as exc:  # noqa: BLE001
                info_domain(
                    "event",
                    "Event failed: scenes list",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="scene_list",
                    error_type=exc.__class__.__name__,
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("не удалось получить список сцен")
                await _send_event_fail_screen(
                    message,
                    state,
                    user_id=user_id,
                    retry_count=retry_count,
                )
                return
            user_gender = data.get("gender")
            if not user_gender:
                profile = await repository.ensure_user(user_id)
                user_gender = profile.gender
                if user_gender:
                    await state.update_data(gender=user_gender)
            allowed_scene_genders = _allowed_scene_genders(user_gender)
            if allowed_scene_genders:
                scenes = [
                    scene
                    for scene in scenes
                    if _normalize_gender(scene.gender) in allowed_scene_genders
                ]
            seen_ids = await repository.list_event_seen_scene_ids(
                user_id, event_key
            )
            candidates = [scene for scene in scenes if scene.scene_id not in seen_ids]
            repeated = False
            if not candidates:
                candidates = list(scenes)
                repeated = True
            rng.shuffle(candidates)
            chosen_scene = None
            scene_download = None
            for idx, scene in enumerate(candidates[: min(len(candidates), 5)], start=1):
                try:
                    download = await fetch_drive_bytes(scene.drive_url, retries=3)
                except Exception as exc:  # noqa: BLE001
                    info_domain(
                        "event",
                        "Scene download failed",
                        stage="EVENT_SELECT_SCENE",
                        user_id=user_id,
                        event_id=event_key,
                        scene_id=scene.scene_id,
                        url=scene.drive_url,
                        retry_count=idx - 1,
                        repeat=repeated,
                        ok=False,
                        error_type=exc.__class__.__name__,
                    )
                    continue
                chosen_scene = scene
                scene_download = download
                info_domain(
                    "event",
                    "Scene selected",
                    stage="EVENT_SELECT_SCENE",
                    user_id=user_id,
                    event_id=event_key,
                    scene_id=scene.scene_id,
                    url=scene.drive_url,
                    user_gender=user_gender,
                    scene_gender=scene.gender,
                    retry_count=idx - 1,
                    repeat=repeated,
                    ok=True,
                )
                break
            if chosen_scene is None or scene_download is None:
                info_domain(
                    "event",
                    "Event failed: scene download",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="scene_select",
                    error_type="scene_download_failed",
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("не удалось скачать сцену")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return
            _event_log_human(
                (
                    "🎭 Запуск генерации: оправа №"
                    f"{frame_model.unique_id} ({frame_model.model_code}), сцена №{chosen_scene.scene_id}"
                ),
                user_id=user_id,
            )
            scene_hash = hashlib.sha256(scene_download.data).hexdigest()[:12]
            if debug_dir and debug_meta:
                scene_ext = scene_download.extension or _extension_from_mime(
                    scene_download.mime
                )
                _safe_write_debug_bytes(
                    debug_dir / f"scene_{rid}.{scene_ext}",
                    scene_download.data,
                    user_id=user_id,
                    label="scene",
                )
                debug_meta.update(
                    {
                        "scene": {
                            "bytes": scene_download.size,
                            "hash": scene_hash,
                            "mime": scene_download.mime,
                            "drive_id": scene_download.drive_id,
                            "scene_id": chosen_scene.scene_id,
                        }
                    }
                )
                debug_meta["scene_id"] = chosen_scene.scene_id
                _write_debug_meta()

            await _edit_progress(msg.PROGRESS_WAIT_GENERATION)

            prompt_hash = hashlib.sha256(
                event_prompt.encode("utf-8")
            ).hexdigest()[:12]
            parts_order = ["face", "frame", "scene", "prompt"]
            payload_kind = "bytes_from_file_id"
            if debug_dir and debug_meta:
                _safe_write_debug_text(
                    debug_dir / f"prompt_{rid}.json",
                    event_prompt,
                    user_id=user_id,
                    label="prompt",
                )
                debug_meta.update(
                    {
                        "prompt": {
                            "len": len(event_prompt),
                            "hash": prompt_hash,
                        },
                        "order": parts_order,
                        "payload_kind": payload_kind,
                    }
                )
                _write_debug_meta()
            info_domain(
                "event",
                "Event NanoBanana manifest",
                stage="EVENT_NANOBANANA_MULTIPART_MANIFEST",
                user_id=user_id,
                event_id=event_key,
                order=parts_order,
                parts=[
                    {"name": "face", "mime": face_mime, "size": len(face_bytes)},
                    {
                        "name": "frame",
                        "mime": frame_download.mime,
                        "size": frame_download.size,
                    },
                    {
                        "name": "scene",
                        "mime": scene_download.mime,
                        "size": scene_download.size,
                    },
                    {"name": "prompt", "mime": "application/json", "size": len(event_prompt)},
                ],
            )
            info_domain(
                "event",
                "Event NanoBanana request",
                stage="EVENT_NANOBANANA_REQUEST",
                user_id=user_id,
                event_id=event_key,
                mode="EVENT",
                face_bytes=len(face_bytes),
                frame_bytes=frame_download.size,
                scene_bytes=scene_download.size,
                face_mime=face_mime,
                frame_mime=frame_download.mime,
                scene_mime=scene_download.mime,
                face_hash=face_hash,
                frame_hash=frame_hash,
                scene_hash=scene_hash,
                frame_drive_id=frame_download.drive_id,
                scene_drive_id=scene_download.drive_id,
                order=parts_order,
                face_source=face_source,
                payload_kind=payload_kind,
                prompt_len=len(event_prompt),
                prompt_hash=prompt_hash,
            )
            start_time = time.perf_counter()
            generation_result = await with_generation_slot(
                generate_event(
                    face_bytes=face_bytes,
                    face_mime=face_mime,
                    glasses_bytes=frame_download.data,
                    glasses_mime=frame_download.mime,
                    scene_bytes=scene_download.data,
                    scene_mime=scene_download.mime,
                    prompt_json=event_prompt,
                    model_name=resolved_event_model_name,
                )
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            generation_latency_ms = latency_ms
            info_domain(
                "event",
                "Event NanoBanana response",
                stage="EVENT_NANOBANANA_RESPONSE",
                user_id=user_id,
                event_id=event_key,
                status_code=200,
                latency_ms=latency_ms,
                ok=True,
            )
            result_bytes = generation_result.image_bytes
            response_len = len(result_bytes)
            response_hash = hashlib.sha256(result_bytes).hexdigest()[:12]
            response_signature = _bytes_signature(result_bytes)
            response_kind = _detect_bytes_kind(result_bytes)
            info_domain(
                "event",
                "Event NanoBanana response bytes",
                stage="EVENT_NANOBANANA_RESPONSE_BYTES",
                user_id=user_id,
                event_id=event_key,
                response_bytes_len=response_len,
                response_hash=response_hash,
                response_signature=response_signature,
            )
            info_domain(
                "event",
                "Event response kind",
                stage="EVENT_RESPONSE_KIND",
                user_id=user_id,
                event_id=event_key,
                response_kind=response_kind,
            )
            if debug_dir and debug_meta:
                _safe_write_debug_bytes(
                    debug_dir / f"response_raw_{rid}.bin",
                    result_bytes,
                    user_id=user_id,
                    label="response_raw",
                )
                debug_meta.update(
                    {
                        "response": {
                            "bytes": response_len,
                            "hash": response_hash,
                            "signature": response_signature,
                            "kind": response_kind,
                        }
                    }
                )
                _write_debug_meta()

            if response_kind not in {"image/png", "image/jpeg", "image/webp"}:
                info_domain(
                    "event",
                    "Event failed: response not image",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="response_validation",
                    error_type="response_not_image",
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("ответ генерации не изображение")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return

            matched_input = None
            if response_hash == face_hash:
                matched_input = "face"
            elif response_hash == frame_hash:
                matched_input = "frame"
            elif response_hash == scene_hash:
                matched_input = "scene"
            if matched_input:
                info_domain(
                    "event",
                    "Event result equals input",
                    stage="EVENT_RESULT_EQUALS_INPUT_BUG",
                    user_id=user_id,
                    event_id=event_key,
                    matched_input=matched_input,
                    response_hash=response_hash,
                )
                info_domain(
                    "event",
                    "Event failed: response equals input",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="response_validation",
                    error_type="response_equals_input",
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("результат совпал с исходным изображением")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return

            saved_bytes = result_bytes
            saved_len = len(saved_bytes)
            saved_hash = hashlib.sha256(saved_bytes).hexdigest()[:12]
            saved_signature = _bytes_signature(saved_bytes)
            saved_match = None
            if saved_hash == face_hash:
                saved_match = "face"
            elif saved_hash == frame_hash:
                saved_match = "frame"
            elif saved_hash == scene_hash:
                saved_match = "scene"
            if saved_match:
                info_domain(
                    "event",
                    "Event result equals input",
                    stage="EVENT_RESULT_EQUALS_INPUT_BUG",
                    user_id=user_id,
                    event_id=event_key,
                    matched_input=saved_match,
                    response_hash=saved_hash,
                )
                info_domain(
                    "event",
                    "Event failed: response equals input",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="response_validation",
                    error_type="response_equals_input",
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("результат совпал с исходным изображением")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return
            results_dir = ensure_dir(Path("./results"))
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            safe_event = event_key.replace(" ", "_")
            result_path = results_dir / f"event_{safe_event}_{user_id}_{timestamp}.png"
            saved_source = "response"
            try:
                result_path.write_bytes(saved_bytes)
            except OSError as exc:
                info_domain(
                    "event",
                    "Event save failed",
                    stage="EVENT_SAVE_RESULT",
                    user_id=user_id,
                    event_id=event_key,
                    ok=False,
                    error_type=exc.__class__.__name__,
                )
                info_domain(
                    "event",
                    "Event failed: save",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="save_result",
                    error_type=exc.__class__.__name__,
                    retryable=True,
                    attempts_not_charged=True,
                )
                _event_log_failure("не удалось сохранить результат")
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return

            info_domain(
                "event",
                "Event saved",
                stage="EVENT_SAVE_RESULT",
                user_id=user_id,
                event_id=event_key,
                ok=True,
                path=str(result_path),
            )
            info_domain(
                "event",
                "Event save result details",
                stage="EVENT_SAVE_RESULT_DETAILS",
                user_id=user_id,
                event_id=event_key,
                saved_bytes_len=saved_len,
                saved_hash=saved_hash,
                saved_signature=saved_signature,
                saved_path=str(result_path),
                source_of_saved_bytes=saved_source,
            )
            if debug_dir and debug_meta:
                saved_ext = _extension_from_mime(response_kind)
                _safe_write_debug_bytes(
                    debug_dir / f"result_saved_{rid}.{saved_ext}",
                    saved_bytes,
                    user_id=user_id,
                    label="result_saved",
                )
                debug_meta.update(
                    {
                        "saved": {
                            "bytes": saved_len,
                            "hash": saved_hash,
                            "signature": saved_signature,
                            "path": str(result_path),
                            "source": saved_source,
                        }
                    }
                )
                _write_debug_meta()
            await _delete_progress()

            if charge_attempts:
                commit_result = await repository.commit_event_attempt(
                    user_id,
                    event_key,
                    rid,
                    scene_id=chosen_scene.scene_id,
                )
                info_domain(
                    "event",
                    "Event attempt commit",
                    stage="EVENT_ATTEMPT_COMMIT",
                    user_id=user_id,
                    event_id=event_key,
                    rid=rid,
                    scene_id=chosen_scene.scene_id,
                    ok=commit_result.ok,
                    status=commit_result.status,
                    use_free=commit_result.use_free,
                    reserved=commit_result.reserved,
                    free_unlocked=commit_result.free_unlocked,
                    free_used=commit_result.free_used,
                    paid_used=commit_result.paid_used,
                )
                if not commit_result.ok:
                    _event_log_failure("не удалось списать попытку")
                    await _send_event_fail_screen(
                        message,
                        state,
                        user_id=user_id,
                        retry_count=retry_count,
                    )
                    return
                attempt_finalized = True
                attempts_left_after_db = _event_attempts_left(
                    phone_present=phone_present,
                    free_unlocked=commit_result.free_unlocked,
                    free_used=commit_result.free_used,
                    paid_used=commit_result.paid_used,
                )
                show_upsell_after = bool(
                    not phone_present
                    and (commit_result.free_used + commit_result.paid_used) == 1
                )
                info_domain(
                    "event",
                    "Event commit success",
                    stage="EVENT_COMMIT_SUCCESS",
                    user_id=user_id,
                    event_id=event_key,
                    scene_id=chosen_scene.scene_id,
                    attempts_left_after=attempts_left_after_db,
                    attempts_left_ui=attempts_left_after_db,
                    use_free=commit_result.use_free,
                    rid=rid,
                )
            else:
                attempts_left_after_db = _event_attempts_left(
                    phone_present=phone_present,
                    free_unlocked=free_unlocked,
                    free_used=free_used,
                    paid_used=paid_used,
                )
                show_upsell_after = bool(
                    not phone_present and (free_used + paid_used) == 0
                )

            data_for_delivery = await state.get_data()
            active_cycle_id = _normalize_cycle(
                data_for_delivery.get("event_active_cycle_id")
            )
            result_site_url = (getattr(frame_model, "site_url", None) or "").strip()
            is_active_delivery = _is_event_cycle_active(
                active_cycle_id, resolved_cycle
            )
            if is_active_delivery:
                caption_text = _event_result_caption(
                    attempts_left=attempts_left_after_db,
                    show_upsell=show_upsell_after,
                )
                result_keyboard = _event_result_keyboard(
                    frame_model, attempts_left=attempts_left_after_db
                )
            else:
                caption_text = ""
                result_keyboard = _event_details_markup_from_url(result_site_url)
            try:
                result_message = await message.answer_photo(
                    BufferedInputFile(saved_bytes, filename="event_result.png"),
                    caption=caption_text,
                    reply_markup=result_keyboard,
                )
            except TelegramBadRequest as exc:
                info_domain(
                    "event",
                    "Event send failed",
                    stage="EVENT_SEND_RESULT",
                    user_id=user_id,
                    event_id=event_key,
                    ok=False,
                    error_type=exc.__class__.__name__,
                    rid=rid,
                )
                info_domain(
                    "event",
                    "Event failed: send",
                    stage="EVENT_FAIL",
                    user_id=user_id,
                    event_id=event_key,
                    stage_name="send_result",
                    error_type=exc.__class__.__name__,
                    retryable=True,
                    attempts_not_charged=False,
                    rid=rid,
                )
                _event_log_failure(
                    "?? ??????? ????????? ?????????",
                    attempts_charged=charge_attempts,
                )
                await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
                return

            info_domain(
                "event",
                "Event result delivered",
                stage="EVENT_SEND_RESULT",
                user_id=user_id,
                event_id=event_key,
                ok=True,
                message_id=result_message.message_id,
                rid=rid,
            )
            await _register_event_session(
                state,
                result_message,
                cycle_id=resolved_cycle,
                photo_context=resolved_context or {},
                site_url=result_site_url or None,
            )
            data_after_send = await state.get_data()
            active_cycle_after_send = _normalize_cycle(
                data_after_send.get("event_active_cycle_id")
            )
            is_active_after_send = _is_event_cycle_active(
                active_cycle_after_send, resolved_cycle
            )
            if is_active_after_send:
                await state.update_data(
                    last_event_result_message_id=result_message.message_id,
                    last_event_result_site_url=result_site_url or None,
                    last_event_result_cycle_id=resolved_cycle,
                )
            else:
                result_chat_id = _resolve_chat_id(result_message) or message.chat.id
                await _cleanup_event_result_message(
                    message.bot,
                    chat_id=int(result_chat_id),
                    message_id=int(result_message.message_id),
                    reason="event_result_stale_post_send",
                    user_id=user_id,
                    site_url=result_site_url or None,
                )
            latency_sec = generation_latency_ms / 1000.0
            _event_log_human(
                (
                    f"✅ Готово за {latency_sec:.1f}s. "
                    f"Осталось попыток: {attempts_left_after_db} "
                    f"(msg_id={result_message.message_id})"
                ),
                user_id=user_id,
            )
            await _reset_event_fail_retry_count(state)

            if show_upsell_after:
                info_domain(
                    "event",
                    "Event phone offer shown",
                    stage="EVENT_PHONE_OFFER_SHOWN",
                    user_id=user_id,
                    event_id=event_key,
                    reason="first_success_event",
                    phone_present=phone_present,
                )
                try:
                    await _send_event_phone_prompt(message, state, user_id)
                except (TelegramBadRequest, TelegramForbiddenError) as exc:
                    info_domain(
                        "event",
                        "Event postprocess Telegram error",
                        stage="EVENT_POSTPROCESS_TG_ERROR",
                        user_id=user_id,
                        event_id=event_key,
                        action="send_phone_prompt",
                        error_type=exc.__class__.__name__,
                    )

        except NanoBananaGenerationError as exc:
            latency_ms = (
                int((time.perf_counter() - start_time) * 1000)
                if start_time
                else 0
            )
            info_domain(
                "event",
                "Event NanoBanana response",
                stage="EVENT_NANOBANANA_RESPONSE",
                user_id=user_id,
                event_id=event_key,
                status_code=exc.status_code,
                latency_ms=latency_ms,
                ok=False,
                reason_code=exc.reason_code,
                reason_detail=exc.reason_detail,
            )
            info_domain(
                "event",
                "Event failed: nanobanana",
                stage="EVENT_FAIL",
                user_id=user_id,
                event_id=event_key,
                stage_name="nanobanana",
                error_type=exc.reason_code,
                retryable=exc.reason_code == "TRANSIENT",
                attempts_not_charged=True,
            )
            reason = "ошибка генерации"
            if exc.reason_code:
                reason = f"ошибка генерации ({exc.reason_code})"
            _event_log_failure(reason)
            await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
        except Exception as exc:  # noqa: BLE001
            info_domain(
                "event",
                "Event failed",
                stage="EVENT_FAIL",
                user_id=user_id,
                event_id=event_key,
                stage_name="exception",
                error_type=exc.__class__.__name__,
                retryable=True,
                attempts_not_charged=True,
            )
            _event_log_failure("неожиданная ошибка")
            await _send_event_fail_screen(message, state, user_id=user_id, retry_count=retry_count)
        finally:
            if (
                user_photo_path
                and user_photo_path.exists()
                and source_kind == "file_id"
            ):
                try:
                    user_photo_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Failed to delete event temp file %s",
                        user_photo_path,
                    )
            if reservation and reservation.ok and not attempt_finalized:
                rollback_result = await repository.rollback_event_attempt(
                    user_id,
                    event_key,
                    rid,
                )
                info_domain(
                    "event",
                    "Event attempt rollback",
                    stage="EVENT_ATTEMPT_ROLLBACK",
                    user_id=user_id,
                    event_id=event_key,
                    rid=rid,
                    ok=rollback_result.ok,
                    status=rollback_result.status,
                    use_free=rollback_result.use_free,
                    reserved=rollback_result.reserved,
                    free_unlocked=rollback_result.free_unlocked,
                    free_used=rollback_result.free_used,
                    paid_used=rollback_result.paid_used,
                )
            if reservation and reservation.ok:
                await _cleanup_event_notices(
                    message.bot,
                    user_id=user_id,
                    max_in_flight=max_in_flight,
                )
            await _delete_progress()
            reset_context(tokens)
            await _release_event_lock(user_id, cycle_id=lock_cycle_id)

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

    async def _start_new_event_cycle(state: FSMContext) -> int:
        data = await state.get_data()
        current_cycle = _normalize_cycle(data.get("event_active_cycle_id")) or 0
        new_cycle = current_cycle + 1
        await state.update_data(
            event_active_cycle_id=new_cycle,
            event_current_cycle=new_cycle,
            last_event_result_message_id=None,
            last_event_result_site_url=None,
            last_event_result_cycle_id=None,
        )
        return new_cycle

    def _is_event_cycle_active(
        active_cycle_id: int | None, cycle_id: int | None
    ) -> bool:
        if active_cycle_id is None or cycle_id is None:
            return True
        return int(active_cycle_id) == int(cycle_id)

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
        vote_payload: Mapping[str, str] | None = None,
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
        await _remember_card_message(
            state,
            message,
            title=model.title,
            trimmed=False,
            vote_payload=vote_payload,
        )


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
            "Генерация пропущена — причина=contact_request",
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
            snapshot = await catalog.snapshot()
            changed, cleared = await repository.sync_catalog_version(
                snapshot.version_hash,
                clear_on_change=clear_on_catalog_change,
            )
            if changed:
                action = "cleared" if cleared else "preserved"
                logger.info(
                    "Catalog version updated to %s, history %s",
                    snapshot.version_hash,
                    action,
                )
            candidates = await catalog.list_by_gender(filters.gender)
        except CatalogError as exc:
            logger.error(
                "Ошибка при парсинге каталога: %s",
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
        exclude = set(exclude_ids or set())
        available = [
            model for model in candidates if model.unique_id not in exclude
        ]
        if not available:
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
                f"Нет доступных моделей — reason=no_models",
                stage="GENERATION_SKIPPED",
                user_id=user_id,
                gender=filters.gender,
            )
            return False
        await style_recommender.ensure_user_styles(
            user_id, (model.style for model in snapshot.models)
        )
        available_styles = [model.style for model in available]
        if batch_size == 2:
            selected_pair = await style_recommender.select_style_pair_for_collage(
                user_id,
                available_styles,
                allow_duplicates=True,
            )
            selected_styles = list(selected_pair)
        else:
            selected_styles = await style_recommender.select_styles_for_collage(
                user_id,
                available_styles,
                n=batch_size,
            )
        fallback_styles = await style_recommender.rank_styles_for_collage(
            user_id, available_styles
        )
        batch = style_recommender.select_models_for_collage(
            available,
            selected_styles,
            n=batch_size,
            fallback_styles=fallback_styles,
        )
        if not batch:
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
                "Нет доступных моделей — reason=no_models",
                stage="GENERATION_SKIPPED",
                user_id=user_id,
                gender=filters.gender,
            )
            return False
        candidate_ids = {model.unique_id for model in available}
        batch_ids = {model.unique_id for model in batch}
        remaining_after_batch = max(len(candidate_ids) - len(batch_ids), 0)
        exhausted = remaining_after_batch <= 0
        if len(selected_styles) == 2:
            canonical_pair = tuple(sorted(selected_styles))
            logger.debug(
                "Selected style pair for user %s: %s (display=%s)",
                user_id,
                canonical_pair,
                selected_styles,
            )
        else:
            logger.debug("Selected styles for user %s: %s", user_id, selected_styles)
        logger.debug(
            "Selected models for user %s: %s",
            user_id,
            [
                (
                    model.unique_id,
                    (model.style or STYLE_UNKNOWN).strip() or STYLE_UNKNOWN,
                )
                for model in batch
            ],
        )
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
        if exhausted:
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
        active_mode = _get_active_mode(data)
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
            if active_mode == ACTIVE_MODE_EVENT:
                await state.set_state(TryOnStates.RESULT)
            else:
                await state.set_state(TryOnStates.SHOW_RECS)
        pending_generation = data.get("contact_pending_generation", False)
        allow_generation = (
            send_generation and pending_generation and pending_state != "limit"
        )
        if allow_generation and active_mode != ACTIVE_MODE_EVENT:
            await state.update_data(contact_pending_generation=False)
            filters = await _ensure_filters(user_id, state)
            await _send_models(
                message,
                user_id,
                filters,
                state,
                skip_contact_prompt=True,
            )
        elif allow_generation:
            await state.update_data(contact_pending_generation=False)
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
        active_mode = _get_active_mode(state_data)
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
            if active_mode == ACTIVE_MODE_EVENT:
                data = await state.get_data()
                await _cleanup_last_event_result(
                    message,
                    state,
                    user_id=user_id,
                    reason="event_phone_shared",
                    data=data,
                )
                await _delete_event_aux_message(message, state, data=data)
                await _delete_event_exhausted_message(
                    message, state, data=data, user_id=user_id
                )
                if current_state != TryOnStates.DAILY_LIMIT_REACHED.state:
                    await message.answer(
                        msg.EVENT_PHONE_BONUS_TEXT,
                        reply_markup=event_phone_bonus_keyboard(),
                    )
                return
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
            [(item.unique_id, _format_model_button_label(item)) for item in group],
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


    def _strip_vote_buttons(
        markup: InlineKeyboardMarkup | None,
    ) -> InlineKeyboardMarkup | None:
        if not markup or not markup.inline_keyboard:
            return None
        new_rows = []
        removed = False
        for row in markup.inline_keyboard:
            if any(
                (getattr(button, "callback_data", None) or "").startswith("vote|")
                for button in row
            ):
                removed = True
                continue
            new_rows.append(row)
        if not removed:
            return None
        return InlineKeyboardMarkup(inline_keyboard=new_rows)


    async def _disable_inline_keyboard(
        message: Message | None,
        *,
        reason: str,
        user_id: int | None = None,
        session_key: str | None = None,
        fsm_state: str | None = None,
        allow_delete: bool = True,
    ) -> None:
        if message is None:
            return
        message_id = message.message_id
        edit_exc: Exception | None = None
        try:
            await message.edit_reply_markup(reply_markup=None)
            info_domain(
                "bot.handlers",
                "inline keyboard cleared",
                stage="INLINE_KEYBOARD_CLEARED",
                user_id=user_id,
                message_id=message_id,
                session_key=session_key,
                reason=reason,
                fsm_state=fsm_state,
            )
            return
        except TelegramBadRequest as exc:
            if "message is not modified" in str(exc).lower():
                info_domain(
                    "bot.handlers",
                    "inline keyboard already clear",
                    stage="INLINE_KEYBOARD_ALREADY_CLEAR",
                    user_id=user_id,
                    message_id=message_id,
                    session_key=session_key,
                    reason=reason,
                    fsm_state=fsm_state,
                )
                return
            edit_exc = exc
        except TelegramForbiddenError as exc:
            edit_exc = exc
        except Exception as exc:  # noqa: BLE001
            edit_exc = exc
        if not allow_delete:
            info_domain(
                "bot.handlers",
                "inline keyboard clear failed",
                stage="INLINE_KEYBOARD_CLEAR_FAILED",
                user_id=user_id,
                message_id=message_id,
                session_key=session_key,
                reason=reason,
                fsm_state=fsm_state,
                error=str(edit_exc),
            )
            return
        try:
            await message.delete()
            info_domain(
                "bot.handlers",
                "inline keyboard deleted",
                stage="INLINE_KEYBOARD_DELETED",
                user_id=user_id,
                message_id=message_id,
                session_key=session_key,
                reason=reason,
                fsm_state=fsm_state,
            )
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            info_domain(
                "bot.handlers",
                "inline keyboard clear failed",
                stage="INLINE_KEYBOARD_CLEAR_FAILED",
                user_id=user_id,
                message_id=message_id,
                session_key=session_key,
                reason=reason,
                fsm_state=fsm_state,
                error=str(exc),
            )


    def _append_recent_id(ids: list[str], new_id: str) -> list[str]:
        if new_id in ids:
            ids = [item for item in ids if item != new_id]
        ids.append(new_id)
        if len(ids) > deleted_collage_ids_limit:
            ids = ids[-deleted_collage_ids_limit:]
        return ids


    async def _remember_deleted_collage_id(
        state: FSMContext, lock: asyncio.Lock | None, message_id_str: str
    ) -> None:
        if lock is None:
            data = await state.get_data()
            existing = [str(item) for item in data.get("deleted_collage_ids", [])]
            updated = _append_recent_id(existing, message_id_str)
            await state.update_data(deleted_collage_ids=updated)
            return
        async with lock:
            data = await state.get_data()
            existing = [str(item) for item in data.get("deleted_collage_ids", [])]
            updated = _append_recent_id(existing, message_id_str)
            await state.update_data(deleted_collage_ids=updated)


    async def _remove_collage_message(
        *,
        bot: Bot,
        chat_id: int,
        message_id: int,
        state: FSMContext | None,
        lock: asyncio.Lock | None,
        user_id: int | None,
        session_key: str | None,
        reason: str,
        fsm_state: str | None,
    ) -> None:
        message_id_str = str(message_id)
        if state is not None:
            if lock is None:
                data = await state.get_data()
            else:
                async with lock:
                    data = await state.get_data()
            deleted_ids = {str(item) for item in data.get("deleted_collage_ids", [])}
            if message_id_str in deleted_ids:
                info_domain(
                    "bot.handlers",
                    "collage delete skipped: already deleted",
                    stage="COLLAGE_DELETE_SKIPPED_ALREADY_DELETED",
                    user_id=user_id,
                    message_id=message_id,
                    session_key=session_key,
                    reason=reason,
                    fsm_state=fsm_state,
                )
                return
        info_domain(
            "bot.handlers",
            "collage delete attempt",
            stage="COLLAGE_DELETE_ATTEMPT",
            user_id=user_id,
            message_id=message_id,
            session_key=session_key,
            reason=reason,
            fsm_state=fsm_state,
        )
        try:
            await bot.delete_message(chat_id, message_id)
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            exc_text = str(exc).lower()
            not_found = "message to delete not found" in exc_text
            cannot_delete = "message can't be deleted" in exc_text or "message cannot be deleted" in exc_text
            if not_found:
                info_domain(
                    "bot.handlers",
                    "collage delete not found",
                    stage="COLLAGE_DELETE_NOT_FOUND",
                    user_id=user_id,
                    message_id=message_id,
                    session_key=session_key,
                    reason=reason,
                    fsm_state=fsm_state,
                    error=str(exc),
                )
                if state is not None:
                    await _remember_deleted_collage_id(state, lock, message_id_str)
                return
            if cannot_delete:
                info_domain(
                    "bot.handlers",
                    "collage delete not found",
                    stage="COLLAGE_DELETE_NOT_FOUND",
                    user_id=user_id,
                    message_id=message_id,
                    session_key=session_key,
                    reason=reason,
                    fsm_state=fsm_state,
                    error=str(exc),
                )
            try:
                await bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=message_id, reply_markup=None
                )
            except TelegramBadRequest as edit_exc:
                if "message is not modified" in str(edit_exc).lower():
                    info_domain(
                        "bot.handlers",
                        "collage delete fallback edit ok",
                        stage="COLLAGE_DELETE_FALLBACK_EDIT_OK",
                        user_id=user_id,
                        message_id=message_id,
                        session_key=session_key,
                        reason=reason,
                        fsm_state=fsm_state,
                    )
                else:
                    info_domain(
                        "bot.handlers",
                        "collage delete fallback edit failed",
                        stage="COLLAGE_DELETE_FALLBACK_EDIT_FAILED",
                        user_id=user_id,
                        message_id=message_id,
                        session_key=session_key,
                        reason=reason,
                        fsm_state=fsm_state,
                        error=str(edit_exc),
                    )
            else:
                info_domain(
                    "bot.handlers",
                    "collage delete fallback edit ok",
                    stage="COLLAGE_DELETE_FALLBACK_EDIT_OK",
                    user_id=user_id,
                    message_id=message_id,
                    session_key=session_key,
                    reason=reason,
                    fsm_state=fsm_state,
                )
            if cannot_delete and state is not None:
                await _remember_deleted_collage_id(state, lock, message_id_str)
            return
        except Exception as exc:  # noqa: BLE001
            info_domain(
                "bot.handlers",
                "collage delete failed",
                stage="COLLAGE_DELETE_FAILED",
                user_id=user_id,
                message_id=message_id,
                session_key=session_key,
                reason=reason,
                fsm_state=fsm_state,
                error=str(exc),
            )
            try:
                await bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=message_id, reply_markup=None
                )
            except TelegramBadRequest as edit_exc:
                if "message is not modified" in str(edit_exc).lower():
                    info_domain(
                        "bot.handlers",
                        "collage delete fallback edit ok",
                        stage="COLLAGE_DELETE_FALLBACK_EDIT_OK",
                        user_id=user_id,
                        message_id=message_id,
                        session_key=session_key,
                        reason=reason,
                        fsm_state=fsm_state,
                    )
                    return
                info_domain(
                    "bot.handlers",
                    "collage delete fallback edit failed",
                    stage="COLLAGE_DELETE_FALLBACK_EDIT_FAILED",
                    user_id=user_id,
                    message_id=message_id,
                    session_key=session_key,
                    reason=reason,
                    fsm_state=fsm_state,
                    error=str(edit_exc),
                )
                return
            info_domain(
                "bot.handlers",
                "collage delete fallback edit ok",
                stage="COLLAGE_DELETE_FALLBACK_EDIT_OK",
                user_id=user_id,
                message_id=message_id,
                session_key=session_key,
                reason=reason,
                fsm_state=fsm_state,
            )
            return
        info_domain(
            "bot.handlers",
            "collage delete ok",
            stage="COLLAGE_DELETE_OK",
            user_id=user_id,
            message_id=message_id,
            session_key=session_key,
            reason=reason,
            fsm_state=fsm_state,
        )
        if state is not None:
            await _remember_deleted_collage_id(state, lock, message_id_str)


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
            active_mode=ACTIVE_MODE_WEAR,
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
            "Пользователь открыл старт",
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

    @router.callback_query(StateFilter(TryOnStates.FOR_WHO), F.data.startswith("gender_"))
    async def select_gender(callback: CallbackQuery, state: FSMContext) -> None:
        if not callback.data or not callback.data.startswith("gender_"):
            await callback.answer()
            return
        fsm_state = await state.get_state()
        gender = callback.data.replace("gender_", "")
        await repository.update_filters(callback.from_user.id, gender=gender)
        await state.update_data(
            gender=gender,
            first_generated_today=True,
            allow_try_button=False,
        )
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await track_event(str(callback.from_user.id), "gender_selected", value=gender)
        await _disable_inline_keyboard(
            callback.message,
            reason="gender_selected",
            user_id=callback.from_user.id,
            fsm_state=fsm_state,
            allow_delete=False,
        )
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
            f"Выбор: пол={gender}",
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
        active_mode = _get_active_mode(data_before)
        if active_mode == ACTIVE_MODE_EVENT:
            await _delete_last_aux_message(message, state)
            await _cleanup_last_event_result(
                message,
                state,
                user_id=user_id,
                reason="event_new_photo",
                data=data_before,
            )
            await _delete_event_aux_message(message, state, data=data_before)
            await _delete_event_exhausted_message(
                message, state, data=data_before, user_id=user_id
            )
            phone_present = await _event_phone_present(user_id)
            if not await _event_preflight_attempt(
                message, state, user_id=user_id, phone_present=phone_present
            ):
                return
            photo = message.photo[-1]
            path = await save_user_photo(message)
            event_cycle = await _start_new_event_cycle(state)
            await state.update_data(
                event_upload=path,
                event_upload_file_id=photo.file_id,
                event_last_photo_file_id=photo.file_id,
                event_current_cycle=event_cycle,
                event_fail_retry_count=0,
            )
            photo_context = {
                "upload": path,
                "upload_file_id": photo.file_id,
                "last_photo_file_id": photo.file_id,
            }
            info_domain(
                "bot.handlers",
                "Photo received",
                stage="PHOTO_RECEIVED",
                user_id=user_id,
                active_mode=active_mode,
                saved_as_last_photo=True,
                next_action="event_flow",
            )
            await state.set_state(TryOnStates.RESULT)
            await _run_event_generation(
                message,
                state,
                user_id=user_id,
                source="photo_upload",
                photo_context=photo_context,
                event_cycle_id=event_cycle,
            )
            return
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
        next_action = "event_ack" if active_mode == ACTIVE_MODE_EVENT else "wear_flow"
        info_domain(
            "bot.handlers",
            "Photo received",
            stage="PHOTO_RECEIVED",
            user_id=user_id,
            active_mode=active_mode,
            saved_as_last_photo=True,
            next_action=next_action,
        )
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
                "Достигнут дневной лимит",
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
            "Фото получено",
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

    @router.callback_query(F.data.startswith("vote|"))
    async def vote_on_style(callback: CallbackQuery, state: FSMContext) -> None:
        message = callback.message
        payload = callback.data or ""
        user_id = (
            callback.from_user.id
            if callback.from_user
            else message.chat.id
            if message
            else None
        )
        if message is None or not payload or user_id is None:
            await callback.answer()
            return
        parts = payload.split("|", 4)
        if len(parts) < 4:
            await callback.answer(msg.VOTE_ACK_INVALID)
            return
        _, vote, generation_id, style, *_ = parts
        status = await style_recommender.update_from_vote(
            user_id, generation_id, style, vote
        )
        if status in {VOTE_OK, VOTE_DUPLICATE}:
            stripped = _strip_vote_buttons(message.reply_markup)
            if stripped is not None:
                try:
                    if stripped.inline_keyboard:
                        await message.edit_reply_markup(reply_markup=stripped)
                    else:
                        await message.edit_reply_markup(reply_markup=None)
                except (TelegramBadRequest, TelegramForbiddenError) as exc:
                    logger.debug(
                        "Failed to update vote markup %s: %s",
                        message.message_id,
                        exc,
                    )
            state_data = await state.get_data()
            last_card = state_data.get("last_card_message")
            if (
                isinstance(last_card, Mapping)
                and last_card.get("message_id") == message.message_id
                and last_card.get("vote_payload") is not None
            ):
                updated_card = dict(last_card)
                updated_card.pop("vote_payload", None)
                await state.update_data(last_card_message=updated_card)
            profile = await repository.ensure_user(user_id)
            if (
                profile.last_more_message_id == message.message_id
                and profile.last_more_message_type == "result"
                and profile.last_more_message_payload
            ):
                payload_data = dict(profile.last_more_message_payload)
                if payload_data.get("vote_payload") is not None:
                    payload_data["vote_payload"] = None
                    await repository.set_last_more_message(
                        user_id,
                        profile.last_more_message_id,
                        profile.last_more_message_type,
                        payload_data,
                    )
        if status == VOTE_OK:
            response_text = msg.VOTE_ACK_OK
        elif status == VOTE_DUPLICATE:
            response_text = msg.VOTE_ACK_DUPLICATE
        else:
            response_text = msg.VOTE_ACK_INVALID
        await callback.answer(response_text)

    @router.callback_query(F.data.startswith("pick:"))
    async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
        """Launch generation using the model tied to the tapped collage, honoring its original cycle/photo context."""
        parts = callback.data.split(":", 2)
        if len(parts) == 3:
            _, batch_source_key, model_id = parts
        else:  # fallback for legacy format
            batch_source_key = "unknown"
            model_id = callback.data.replace("pick:", "", 1)
        message = callback.message
        if message is None:
            await callback.answer()
            return
        message_id_str = str(message.message_id)
        user_id = callback.from_user.id if callback.from_user else message.chat.id
        collage_lock = _get_collage_lock(user_id)
        claimed = False
        repeat_reason = None
        data: dict[str, Any] = {}
        used_message_ids: set[str] = set()
        pending_collage_ids: set[str] = set()
        fsm_state = None
        async with collage_lock:
            data = await state.get_data()
            fsm_state = await state.get_state()
            result_messages = dict(data.get("result_messages", {}))
            used_message_ids = {str(key) for key in data.get("used_message_ids", [])}
            used_message_ids.update(result_messages.keys())
            pending_collage_ids = {str(key) for key in data.get("pending_collage_ids", [])}
            if message_id_str in pending_collage_ids:
                repeat_reason = "pending"
            elif message_id_str in used_message_ids:
                repeat_reason = "used"
            else:
                pending_collage_ids.add(message_id_str)
                await state.update_data(pending_collage_ids=list(pending_collage_ids))
                claimed = True
        models_msg_id = data.get("models_message_id")
        sessions = dict(data.get("collage_sessions", {}))
        session_key = message_id_str
        session_entry = sessions.get(session_key)
        if not session_entry:
            for key, entry in sessions.items():
                aliases = {str(alias) for alias in entry.get("aliases", [])}
                if message_id_str in aliases:
                    session_key = key
                    session_entry = entry
                    break
        info_domain(
            "bot.handlers",
            "choose_model: callback received",
            stage="CHOOSE_MODEL_START",
            user_id=user_id,
            message_id=message.message_id,
            callback_data=callback.data,
            models_msg_id=models_msg_id,
            session_key=session_key,
            has_session_entry=bool(session_entry),
            collage_session_keys=list(sessions.keys()),
            used_message_ids=list(used_message_ids),
            pending_collage_ids=list(pending_collage_ids),
            fsm_state=fsm_state,
        )
        if repeat_reason:
            info_domain(
                "bot.handlers",
                "choose_model: repeated click ignored",
                stage="CHOOSE_MODEL_REPEATED_CLICK_IGNORED",
                user_id=user_id,
                message_id=message.message_id,
                session_key=session_key,
                reason=repeat_reason,
                fsm_state=fsm_state,
            )
            await _remove_collage_message(
                bot=message.bot,
                chat_id=message.chat.id,
                message_id=message.message_id,
                user_id=user_id,
                session_key=session_key,
                reason=f"repeat_{repeat_reason}",
                fsm_state=fsm_state,
                state=state,
                lock=collage_lock,
            )
            await callback.answer()
            return
        info_domain(
            "bot.handlers",
            "choose_model: collage claimed",
            stage="CHOOSE_MODEL_CLAIMED",
            user_id=user_id,
            message_id=message.message_id,
            session_key=session_key,
            fsm_state=fsm_state,
        )
        await _remove_collage_message(
            bot=message.bot,
            chat_id=message.chat.id,
            message_id=message.message_id,
            user_id=user_id,
            session_key=session_key,
            reason="claimed",
            fsm_state=fsm_state,
            state=state,
            lock=collage_lock,
        )
        sessions_after: dict[str, Any] = {}
        final_state = None
        try:
            if models_msg_id and models_msg_id == message.message_id:
                try:
                    await message.bot.delete_message(message.chat.id, models_msg_id)
                except Exception:
                    pass
                await state.update_data(models_message_id=None)

            if not session_entry and not (models_msg_id and message.message_id == models_msg_id):
                info_domain(
                    "bot.handlers",
                    "choose_model: model unavailable",
                    stage="CHOOSE_MODEL_MODEL_UNAVAILABLE",
                    user_id=user_id,
                    message_id=message.message_id,
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
            elif models_msg_id and message.message_id == models_msg_id:
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
                    "choose_model: model unavailable",
                    stage="CHOOSE_MODEL_MODEL_UNAVAILABLE",
                    user_id=user_id,
                    message_id=message.message_id,
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
                    "choose_model: model unavailable",
                    stage="CHOOSE_MODEL_MODEL_UNAVAILABLE",
                    user_id=user_id,
                    message_id=message.message_id,
                    callback_data=callback.data,
                    models_msg_id=models_msg_id,
                    has_session_entry=bool(session_entry),
                )
                await callback.answer(msg.MODEL_UNAVAILABLE_ALERT, show_alert=True)
                return
            logger.debug(
                "User %s selected model %s from %s",
                user_id,
                model_id,
                batch_source_key,
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
                    "daily_limit_after_pick",
                    stage="DAILY_LIMIT",
                    user_id=user_id,
                    context="model_pick",
                )
                await callback.answer()
                return
            await callback.answer()
            if generation_cycle is None:
                generation_cycle = await _ensure_current_cycle_id(state, user_id)
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
            info_domain(
                "bot.handlers",
                "choose_model: launch generation",
                stage="CHOOSE_MODEL_LAUNCH_GENERATION",
                user_id=user_id,
                message_id=message.message_id,
                model_id=model_id,
                generation_cycle=generation_cycle,
                session_key=session_key,
                has_session_entry=bool(session_entry),
            )
            await _perform_generation(
                message,
                state,
                selected,
                generation_cycle=generation_cycle,
                photo_context=photo_context,
            )
        finally:
            if claimed:
                async with collage_lock:
                    data_after = await state.get_data()
                    final_state = await state.get_state()
                    pending_after = {str(key) for key in data_after.get("pending_collage_ids", [])}
                    pending_after.discard(message_id_str)
                    used_after = {str(key) for key in data_after.get("used_message_ids", [])}
                    used_after.add(message_id_str)
                    sessions_after = dict(data_after.get("collage_sessions", {}))
                    sessions_after.pop(session_key, None)
                    await state.update_data(
                        pending_collage_ids=list(pending_after),
                        used_message_ids=list(used_after),
                        collage_sessions=sessions_after,
                        models_message_id=None,
                    )
                    info_domain(
                        "bot.handlers",
                        "choose_model: pending/used updated",
                        stage="CHOOSE_MODEL_PENDING_USED_UPDATED",
                        user_id=user_id,
                        message_id=message.message_id,
                        session_key=session_key,
                        fsm_state=final_state,
                        pending_count=len(pending_after),
                        used_count=len(used_after),
                    )
                await _remove_collage_message(
                    bot=message.bot,
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    user_id=user_id,
                    session_key=session_key,
                    reason="finalize",
                    fsm_state=final_state,
                    state=state,
                    lock=collage_lock,
                )
                info_domain(
                    "bot.handlers",
                    "choose_model: collage session cleared",
                    stage="CHOOSE_MODEL_COLLAGE_SESSION_CLEARED",
                    user_id=user_id,
                    message_id=message.message_id,
                    cleared_session_key=session_key,
                    remaining_collage_keys=list(sessions_after.keys()),
                    fsm_state=final_state,
                )


    @router.callback_query(
        StateFilter(ContactRequest.waiting_for_phone),
        F.data == CONTACT_SHARE_CALLBACK,
    )
    async def contact_share_button(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="contact_share")
        message = callback.message
        if message is None:
            return
        await _maybe_delete_event_attempts_message(message, state, user_id)
        await _delete_contact_prompt_message(message, state)
        prompt_message = await message.answer(
            msg.ASK_PHONE_PROMPT_MANUAL,
            reply_markup=contact_share_reply_keyboard(),
        )
        await state.update_data(
            contact_prompt_message_id=prompt_message.message_id,
            last_aux_message_id=prompt_message.message_id,
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
            "_perform_generation: старт генерации",
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
                f"Генерация запущена — frame={model.unique_id}",
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
                f"Генерация готова — {latency_ms} ms",
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
        clean_style = (
            (getattr(model, "style", None) or STYLE_UNKNOWN).strip() or STYLE_UNKNOWN
        )
        generation_id = f"{generation_cycle}:{model.unique_id}"
        vote_payload = {
            "generation_id": generation_id,
            "style": clean_style,
            "model_id": model.unique_id,
        }
        await _delete_progress_message()
        info_domain(
            "bot.handlers",
            "_perform_generation: статус цикла перед доставкой результата",
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
                vote_payload=vote_payload,
            )
            info_domain(
                "bot.handlers",
                "_perform_generation: отправляем stale-результат",
                stage="GEN_DELIVER_STALE",
                user_id=user_id,
                generation_cycle=generation_cycle,
                model_id=model.unique_id,
            )
            stale_message = await _send_delivery_message(
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
            await _handle_wear_success_event(
                message,
                user_id=user_id,
                generation_cycle=generation_cycle,
                generation_id=generation_id,
                source_message_id=message.message_id,
                result_message_id=stale_message.message_id,
                photo_file_id=upload_file_id or last_photo_file_id,
                model_id=model.unique_id,
                source="stale",
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
            vote_payload=vote_payload,
        )
        await _deactivate_previous_more_button(message.bot, user_id)
        info_domain(
            "bot.handlers",
            "_perform_generation: отправляем актуальный результат",
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
            "vote_payload": vote_payload,
        })
        # === /SAVE last_card_message ===

        info_domain(
            "bot.handlers",
            "_perform_generation: регистрируем результат и alias по source_message_id",
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
            vote_payload=vote_payload,
        )
        await _handle_wear_success_event(
            message,
            user_id=user_id,
            generation_cycle=generation_cycle,
            generation_id=generation_id,
            source_message_id=message.message_id,
            result_message_id=result_message.message_id,
            photo_file_id=upload_file_id or last_photo_file_id,
            model_id=model.unique_id,
            source="current",
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
                {"site_url": model.site_url, "vote_payload": vote_payload},
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
                "Достигнут дневной лимит",
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
        active_mode = _get_active_mode(data_before)
        if active_mode == ACTIVE_MODE_EVENT:
            await callback.answer()
            return
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
                "Достигнут дневной лимит",
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
                "Достигнут дневной лимит",
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
                "Достигнут дневной лимит",
                stage="DAILY_LIMIT",
                user_id=user_id,
                context=context,
            )
            return
        prompt_text = _render_text(msg.PHOTO_INSTRUCTION)
        await _prompt_for_next_photo(message, state, prompt_text)

    @router.message(Command("wear"))
    async def command_wear(
        message: Message,
        state: FSMContext,
        *,
        source: str = "command",
        user_id_override: int | None = None,
    ) -> None:
        if user_id_override is not None:
            user_id = user_id_override
        else:
            if not message.from_user:
                return
            user_id = message.from_user.id
        await _set_active_mode(state, user_id, ACTIVE_MODE_WEAR, source=source)
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="wear_command",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
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
                "Достигнут дневной лимит",
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

    @router.message(Command("event"))
    async def command_event(message: Message, state: FSMContext) -> None:
        if not message.from_user:
            return
        user_id = message.from_user.id
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=message.from_user.id,
            source="command",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="command")
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_command",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        teaser_result = await _maybe_send_event_teaser(
            message,
            user_id,
            source="command",
            send_teaser=True,
        )
        prompt_text = None
        if teaser_result.status != EVENT_TEASER_ALREADY_SENT:
            prompt_text = msg.EVENT_NEED_PHOTO_FROM_TEASER
        seeded_context, _ = await _seed_event_photo_from_wear(
            state, user_id=user_id
        )
        new_cycle = await _start_new_event_cycle(state)
        await _run_event_generation(
            message,
            state,
            user_id=user_id,
            source="command",
            photo_context=seeded_context,
            event_cycle_id=new_cycle,
            photo_prompt_text=prompt_text,
        )

    @router.callback_query(F.data == EVENT_TRY_CALLBACK)
    async def event_trigger(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="trigger")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="trigger",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_trigger",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        await _deactivate_previous_more_button(message.bot, user_id)
        await _trim_last_card_message(message, state, site_url=site_url)
        try:
            await message.bot.delete_message(message.chat.id, message.message_id)
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            logger.debug(
                "Failed to delete event trigger message %s: %s",
                message.message_id,
                exc,
            )
        info_domain(
            "event",
            "Event try callback received",
            stage="EVENT_TRY_CALLBACK_RECEIVED",
            user_id=user_id,
            event_id=event_key,
            source="trigger",
        )
        phone_present = await _event_phone_present(user_id)
        attempts_snapshot = await _event_attempts_snapshot(user_id, phone_present)
        free_unlocked, free_used, paid_used, _, _, attempts_left = attempts_snapshot
        info_domain(
            "event",
            "Event access after unlock",
            stage="EVENT_ACCESS_AFTER_UNLOCK",
            user_id=user_id,
            event_id=event_key,
            source="trigger",
            free_unlocked=free_unlocked,
            free_used=free_used,
            paid_used=paid_used,
            phone_present=phone_present,
            attempts_left=attempts_left,
        )
        if attempts_left <= 0:
            info_domain(
                "event",
                "Event access mismatch",
                stage="EVENT_ACCESS_MISMATCH",
                user_id=user_id,
                event_id=event_key,
                source="trigger",
                db_snapshot={
                    "free_unlocked": free_unlocked,
                    "free_used": free_used,
                    "paid_used": paid_used,
                    "attempts_left": attempts_left,
                    "phone_present": phone_present,
                },
            )
            await message.answer(msg.EVENT_ACCESS_FAILED)
            return
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="callback")
        seeded_context, _ = await _seed_event_photo_from_wear(
            state, user_id=user_id
        )
        new_cycle = await _start_new_event_cycle(state)
        await _run_event_generation(
            message,
            state,
            user_id=user_id,
            source="trigger",
            attempts_snapshot=attempts_snapshot,
            phone_present=phone_present,
            photo_context=seeded_context,
            event_cycle_id=new_cycle,
            photo_prompt_text=msg.EVENT_NEED_PHOTO_FROM_TEASER,
        )

    @router.callback_query(F.data == EVENT_MORE_CALLBACK)
    async def event_more(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="more")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="more",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_more",
            data=data,
            message_ref=message,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        active_cycle_id = _normalize_cycle(data.get("event_active_cycle_id"))
        last_result_id = data.get("last_event_result_message_id")
        if last_result_id is None:
            if active_cycle_id is not None:
                return
        else:
            try:
                last_result_id = int(last_result_id)
            except (TypeError, ValueError):
                return
            if last_result_id != message.message_id:
                return
            last_result_cycle = _normalize_cycle(data.get("last_event_result_cycle_id"))
            if (
                last_result_cycle is not None
                and active_cycle_id is not None
                and last_result_cycle != active_cycle_id
            ):
                return
        new_cycle = await _start_new_event_cycle(state)
        session_entry = dict(data.get("event_sessions", {})).get(
            str(message.message_id)
        )
        photo_context = None
        if session_entry:
            photo_context = {
                "upload": session_entry.get("upload"),
                "upload_file_id": session_entry.get("upload_file_id"),
                "last_photo_file_id": session_entry.get("last_photo_file_id"),
            }
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="callback")
        await _run_event_generation(
            message,
            state,
            user_id=user_id,
            source="command",
            photo_context=photo_context,
            event_cycle_id=new_cycle,
        )

    @router.callback_query(F.data == EVENT_FAIL_RETRY_CALLBACK)
    async def event_fail_retry(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="fail_retry")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="fail_retry",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_fail_retry",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        try:
            await message.bot.delete_message(message.chat.id, message.message_id)
        except (TelegramBadRequest, TelegramForbiddenError):
            try:
                await message.edit_reply_markup(reply_markup=None)
            except (TelegramBadRequest, TelegramForbiddenError):
                try:
                    await message.edit_text(msg.EVENT_STARTING)
                except (TelegramBadRequest, TelegramForbiddenError):
                    pass
        retry_count = _event_fail_retry_count(data)
        if retry_count >= EVENT_FAIL_RETRY_MAX:
            await _send_event_fail_screen(
                message,
                state,
                user_id=user_id,
                retry_count=retry_count,
            )
            return
        next_retry = retry_count + 1
        await state.update_data(event_fail_retry_count=next_retry)
        info_domain(
            "event",
            "Event fail retry requested",
            stage="EVENT_FAIL_RETRY",
            user_id=user_id,
            event_id=event_key,
            retry_count=next_retry,
            attempts_charged=False,
        )
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="callback")
        data = await state.get_data()
        photo_context = _extract_event_photo_context(data)
        if not _has_photo_context(photo_context):
            await _prompt_event_photo(message, state, user_id=user_id, source="callback")
            return
        new_cycle = await _start_new_event_cycle(state)
        await _store_event_photo_context(state, photo_context or {}, cycle_id=new_cycle)
        await state.set_state(TryOnStates.RESULT)
        await _run_event_generation(
            message,
            state,
            user_id=user_id,
            source="fail_retry",
            photo_context=dict(photo_context or {}),
            event_cycle_id=new_cycle,
            charge_attempts=False,
        )

    @router.callback_query(F.data == EVENT_FAIL_NEW_PHOTO_CALLBACK)
    async def event_fail_new_photo(
        callback: CallbackQuery, state: FSMContext
    ) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="fail_new_photo")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="fail_new_photo",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_fail_new_photo",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        try:
            await message.bot.delete_message(message.chat.id, message.message_id)
        except (TelegramBadRequest, TelegramForbiddenError):
            try:
                await message.edit_reply_markup(reply_markup=None)
            except (TelegramBadRequest, TelegramForbiddenError):
                try:
                    await message.edit_text(msg.EVENT_NEW_PHOTO_PROMPT)
                except (TelegramBadRequest, TelegramForbiddenError):
                    pass
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="callback")
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.EVENT_NEW_PHOTO_PROMPT,
        )

    @router.callback_query(F.data == EVENT_REUSE_PHOTO_CALLBACK)
    async def event_reuse_photo(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="reuse_photo")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="reuse_photo",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_reuse_photo",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        try:
            await message.bot.delete_message(message.chat.id, message.message_id)
        except (TelegramBadRequest, TelegramForbiddenError):
            try:
                await message.edit_reply_markup(reply_markup=None)
            except (TelegramBadRequest, TelegramForbiddenError):
                try:
                    await message.edit_text("Запускаю генерацию...")
                except (TelegramBadRequest, TelegramForbiddenError):
                    pass
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="callback")
        phone_present = await _event_phone_present(user_id)
        if not await _event_preflight_attempt(
            message, state, user_id=user_id, phone_present=phone_present
        ):
            return
        data = await state.get_data()
        photo_context = _extract_event_photo_context(data)
        if not _has_photo_context(photo_context):
            await _prompt_event_photo(message, state, user_id=user_id, source="callback")
            return
        new_cycle = await _start_new_event_cycle(state)
        await _store_event_photo_context(state, photo_context or {}, cycle_id=new_cycle)
        await state.set_state(TryOnStates.RESULT)
        await _run_event_generation(
            message,
            state,
            user_id=user_id,
            source="reuse_photo",
            photo_context=dict(photo_context or {}),
            event_cycle_id=new_cycle,
        )

    @router.callback_query(F.data == EVENT_NEW_PHOTO_CALLBACK)
    async def event_send_new_photo(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="new_photo")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="new_photo",
        ):
            return
        if not event_enabled or not event_ready:
            await message.answer(msg.EVENT_DISABLED)
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_new_photo",
            data=data,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        try:
            await message.bot.delete_message(message.chat.id, message.message_id)
        except (TelegramBadRequest, TelegramForbiddenError):
            try:
                await message.edit_reply_markup(reply_markup=None)
            except (TelegramBadRequest, TelegramForbiddenError):
                try:
                    await message.edit_text("Ок, жду новое фото.")
                except (TelegramBadRequest, TelegramForbiddenError):
                    pass
        await _set_active_mode(state, user_id, ACTIVE_MODE_EVENT, source="callback")
        phone_present = await _event_phone_present(user_id)
        if not await _event_preflight_attempt(
            message, state, user_id=user_id, phone_present=phone_present
        ):
            return
        await state.set_state(TryOnStates.AWAITING_PHOTO)
        await _send_aux_message(
            message,
            state,
            message.answer,
            msg.EVENT_NEW_PHOTO_PROMPT,
        )

    @router.callback_query(F.data == EVENT_BACK_CALLBACK)
    async def event_back_to_wear(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id
        await _safe_answer_callback(callback, user_id=user_id, source="back")
        message = callback.message
        if message is None:
            return
        bot_id = message.bot.id
        if not _log_event_user_id_sanity(
            user_id=user_id,
            bot_id=bot_id,
            chat_id=message.chat.id,
            from_user_id=callback.from_user.id,
            source="back",
        ):
            return
        data = await state.get_data()
        await _cleanup_last_event_result(
            message,
            state,
            user_id=user_id,
            reason="event_back_to_wear",
            data=data,
            message_ref=message,
        )
        await _delete_event_aux_message(message, state, data=data)
        await _delete_event_exhausted_message(
            message, state, data=data, user_id=user_id
        )
        await _maybe_delete_event_attempts_message(message, state, user_id)
        await command_wear(
            message,
            state,
            source="callback",
            user_id_override=user_id,
        )

    @router.message(Command("admin"))
    async def command_admin(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not is_admin(user_id):
            return
        url = (router.admin_webapp_url or "").strip()
        if not url:
            await message.answer("Admin panel is not configured.")
            return
        markup = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Open admin",
                        web_app=WebAppInfo(url=url),
                    )
                ]
            ]
        )
        await message.answer("Open admin panel.", reply_markup=markup)

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
        user_id = message.from_user.id
        await _set_active_mode(state, user_id, ACTIVE_MODE_WEAR, source="command")
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

    @router.callback_query(F.data == "details_click")
    async def handle_details_click(callback: CallbackQuery) -> None:
        """Handle click on 'Подробнее о модели' button - track click and show site link."""
        await track_event(str(callback.from_user.id), "details_click")
        # Increment site clicks counter in users table
        await repository.increment_site_clicks(callback.from_user.id)
        sanitized = (site_url or "").strip()
        if not sanitized:
            await callback.answer("Ссылка недоступна", show_alert=True)
            return
        # Show button with direct link to the site
        follow_markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text=msg.DETAILS_BUTTON_TEXT, url=sanitized)]]
        )
        await callback.message.answer("🔗 Перейти на сайт:", reply_markup=follow_markup)
        await callback.answer()

    @router.callback_query(F.data == "cta_book")
    async def handle_cta(callback: CallbackQuery) -> None:
        await track_event(str(callback.from_user.id), "cta_book_opened")
        # Increment site clicks counter in users table
        await repository.increment_site_clicks(callback.from_user.id)
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