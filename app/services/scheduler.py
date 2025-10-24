"""Simple reminder scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from typing import Optional

from aiogram import Bot

from app.logging_conf import EVENT_ID
from app.services.repository import Repository

import logging


class ReminderScheduler:
    """Background scheduler sending reminder messages."""

    def __init__(
        self,
        bot: Bot,
        repository: Repository,
        message_text: str,
        keyboard_factory,
        interval_seconds: int = 60,
    ) -> None:
        self._bot = bot
        self._repository = repository
        self._message_text = message_text
        self._keyboard_factory = keyboard_factory
        self._interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._logger = logging.getLogger("loop_bot.scheduler")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    async def stop(self) -> None:
        if self._task:
            self._stop_event.set()
            await self._task
            self._task = None
            self._stop_event.clear()

    async def _runner(self) -> None:
        while not self._stop_event.is_set():
            await self._check_and_send()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    async def _check_and_send(self) -> None:
        now = datetime.now(timezone.utc)
        reminders = await self._repository.list_due_reminders(now)
        for profile in reminders:
            await self._repository.set_reminder(profile.user_id, None)
            await self._bot.send_message(
                chat_id=profile.user_id,
                text=self._message_text,
                reply_markup=self._keyboard_factory(),
            )
