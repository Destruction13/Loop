"""Background service sending social media advertisement messages."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from app.keyboards import social_ad_keyboard
from app.services.repository import Repository
from app.texts import messages as msg


class SocialAdService:
    """Send a one-off social media advertisement after prolonged inactivity."""

    def __init__(
        self,
        *,
        bot: Bot,
        repository: Repository,
        instagram_url: str,
        tiktok_url: str,
        timeout_minutes: int,
        interval_seconds: int = 30,
    ) -> None:
        self._bot = bot
        self._repository = repository
        self._instagram_url = instagram_url
        self._tiktok_url = tiktok_url
        self._timeout_seconds = max(timeout_minutes * 60, 0)
        self._interval = max(interval_seconds, 1)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._logger = logging.getLogger("loop_bot.social_ad")

    def start(self) -> None:
        """Start the background check loop."""

        if self._timeout_seconds <= 0:
            return
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    async def stop(self) -> None:
        """Stop the background loop."""

        if self._task is None:
            return
        self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop_event.clear()

    async def _runner(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._check_and_send()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Social ad check failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    async def _check_and_send(self) -> None:
        if self._timeout_seconds <= 0:
            return
        threshold_ts = int(time.time()) - self._timeout_seconds
        if threshold_ts <= 0:
            return
        candidates = await self._repository.list_social_ad_candidates(threshold_ts)
        if not candidates:
            return
        for profile in candidates:
            await self._send_ad(profile.user_id)

    async def _send_ad(self, user_id: int) -> None:
        keyboard = social_ad_keyboard(self._instagram_url, self._tiktok_url)
        text = f"<b>{msg.SOCIAL_AD_TITLE}</b>\n{msg.SOCIAL_AD_BODY}"
        try:
            await self._bot.send_message(
                chat_id=user_id,
                text=text,
                reply_markup=keyboard,
            )
        except (TelegramForbiddenError, TelegramBadRequest) as exc:
            self._logger.warning("Failed to deliver social ad to %s: %s", user_id, exc)
        except Exception:
            self._logger.exception("Unexpected error sending social ad to %s", user_id)
        finally:
            await self._repository.mark_social_ad_shown(user_id)
