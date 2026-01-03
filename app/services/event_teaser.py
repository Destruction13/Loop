"""Helpers for delivering the event teaser."""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.types import InlineKeyboardMarkup
from aiogram.types.input_file import FSInputFile

from app.services.repository import Repository

EVENT_TEASER_SENT = "sent"
EVENT_TEASER_ALREADY_SENT = "already_sent"
EVENT_TEASER_FAILED = "failed"
EVENT_TEASER_RATE_LIMITED = "rate_limited"
EVENT_TEASER_SKIPPED = "skipped"


@dataclass(frozen=True)
class EventTeaserResult:
    status: str
    claimed: bool
    unlocked: bool
    error_type: str | None = None
    retry_after: float | None = None


@dataclass(frozen=True)
class _SendResult:
    status: str
    error_type: str | None = None
    retry_after: float | None = None


async def maybe_send_event_teaser(
    *,
    bot: Bot,
    repository: Repository,
    user_id: int,
    event_id: str,
    text: str,
    reply_markup: InlineKeyboardMarkup,
    image_path: Path | None = None,
    chat_id: int | None = None,
    send_teaser: bool = True,
    logger: Logger | None = None,
) -> EventTeaserResult:
    if not event_id:
        return EventTeaserResult(
            status=EVENT_TEASER_SKIPPED, claimed=False, unlocked=False
        )
    unlocked = await repository.unlock_event_access(user_id, event_id)
    if not send_teaser:
        return EventTeaserResult(
            status=EVENT_TEASER_SKIPPED, claimed=False, unlocked=unlocked
        )
    claimed = await repository.claim_event_trigger(user_id, event_id)
    if not claimed:
        return EventTeaserResult(
            status=EVENT_TEASER_ALREADY_SENT, claimed=False, unlocked=unlocked
        )
    target_chat = chat_id if chat_id is not None else user_id
    send_result = await _send_event_teaser_message(
        bot=bot,
        chat_id=target_chat,
        text=text,
        reply_markup=reply_markup,
        image_path=image_path,
        logger=logger,
    )
    if send_result.status == EVENT_TEASER_SENT:
        return EventTeaserResult(
            status=EVENT_TEASER_SENT, claimed=True, unlocked=unlocked
        )
    await repository.release_event_trigger(user_id, event_id)
    return EventTeaserResult(
        status=send_result.status,
        claimed=True,
        unlocked=unlocked,
        error_type=send_result.error_type,
        retry_after=send_result.retry_after,
    )


async def _send_event_teaser_message(
    *,
    bot: Bot,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup,
    image_path: Path | None,
    logger: Logger | None,
) -> _SendResult:
    if image_path is not None:
        if image_path.exists():
            try:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=FSInputFile(image_path),
                    caption=text,
                    reply_markup=reply_markup,
                )
                return _SendResult(status=EVENT_TEASER_SENT)
            except TelegramRetryAfter as exc:
                return _SendResult(
                    status=EVENT_TEASER_RATE_LIMITED,
                    error_type=exc.__class__.__name__,
                    retry_after=float(exc.retry_after),
                )
            except (OSError, TelegramBadRequest, TelegramForbiddenError) as exc:
                if logger:
                    logger.warning(
                        "Failed to send event teaser image %s: %s", image_path, exc
                    )
        elif logger:
            logger.warning("Event teaser image missing: %s", image_path)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
        )
        return _SendResult(status=EVENT_TEASER_SENT)
    except TelegramRetryAfter as exc:
        return _SendResult(
            status=EVENT_TEASER_RATE_LIMITED,
            error_type=exc.__class__.__name__,
            retry_after=float(exc.retry_after),
        )
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        if logger:
            logger.debug("Failed to send event teaser: %s", exc)
        return _SendResult(
            status=EVENT_TEASER_FAILED, error_type=exc.__class__.__name__
        )
