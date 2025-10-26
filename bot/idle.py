from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

import messages
from bot import keyboards
from db.init import Database

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IdleConfig:
    ecom_timeout: float
    social_timeout: float
    ecom_url: str
    ecom_button_text: str
    more_button_text: str
    socials_text: str
    socials_buttons: list[tuple[str, str]]


class IdleWatcher:
    def __init__(self, bot: Bot, db: Database, config: IdleConfig) -> None:
        self._bot = bot
        self._db = db
        self._config = config
        self._tasks: dict[int, list[asyncio.Task[None]]] = {}
        self._lock = asyncio.Lock()

    async def touch(self, tg_id: int) -> None:
        await self._db.update_last_activity(tg_id)
        async with self._lock:
            await self._cancel_tasks(tg_id)
            tasks: list[asyncio.Task[None]] = []
            tasks.append(asyncio.create_task(self._schedule_ecom(tg_id)))
            tasks.append(asyncio.create_task(self._schedule_social(tg_id)))
            self._tasks[tg_id] = tasks

    async def _schedule_ecom(self, tg_id: int) -> None:
        try:
            await asyncio.sleep(self._config.ecom_timeout)
            session = await self._db.get_session(tg_id)
            if session is None or session.ecom_prompt_sent:
                return
            caption = random.choice(list(messages.IDLE_ECOM_CAPTIONS))
            message_text = f"{messages.CTA_MESSAGE_TEXT}\n\n{caption}"
            keyboard = keyboards.ecommerce_cta_keyboard(
                self._config.ecom_button_text,
                self._config.ecom_url,
                self._config.more_button_text,
            )
            try:
                await self._bot.send_message(tg_id, message_text, reply_markup=keyboard)
            except (TelegramBadRequest, TelegramForbiddenError) as err:
                LOGGER.warning("Failed to send e-commerce prompt to %s: %s", tg_id, err)
                return
            await self._db.mark_ecom_prompt_sent(tg_id)
            await self._db.log_event(
                tg_id,
                "idle_push_ecom_sent",
                {"message": caption},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected error in e-commerce idle watcher: %s", exc)

    async def _schedule_social(self, tg_id: int) -> None:
        try:
            await asyncio.sleep(self._config.social_timeout)
            session = await self._db.get_session(tg_id)
            if session is None or session.social_ad_sent:
                return
            caption = random.choice(list(messages.SOCIALS_CAPTIONS))
            keyboard = keyboards.socials_keyboard(self._config.socials_buttons)
            try:
                await self._bot.send_message(
                    tg_id,
                    f"{caption}\n\n{self._config.socials_text}",
                    reply_markup=keyboard,
                )
            except (TelegramBadRequest, TelegramForbiddenError) as err:
                LOGGER.warning("Failed to send socials prompt to %s: %s", tg_id, err)
                return
            await self._db.mark_social_ad_sent(tg_id)
            await self._db.log_event(tg_id, "social_ad_sent", {"message": caption})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected error in social idle watcher: %s", exc)

    async def _cancel_tasks(self, tg_id: int) -> None:
        existing = self._tasks.pop(tg_id, [])
        for task in existing:
            task.cancel()
        for task in existing:
            try:
                await task
            except asyncio.CancelledError:
                continue

    async def stop(self) -> None:
        async with self._lock:
            keys = list(self._tasks.keys())
        for tg_id in keys:
            await self._cancel_tasks(tg_id)
