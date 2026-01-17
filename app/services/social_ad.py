"""Background service sending social media advertisement messages."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Sequence

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from app.analytics import track_event
from app.keyboards import more_buttonless_markup, social_ad_keyboard
from app.services.repository import Repository
from app.texts import messages as msg


class SocialAdService:
    """Send a one-off social media advertisement after prolonged inactivity."""

    def __init__(
        self,
        *,
        bot: Bot,
        repository: Repository,
        social_links: Sequence[tuple[str, str]],
        timeout_minutes: int,
        interval_seconds: int = 30,
        tracking_url_getter: Optional[callable] = None,
    ) -> None:
        self._bot = bot
        self._repository = repository
        self._social_links: list[tuple[str, str]] = [
            (title.strip(), url.strip())
            for title, url in social_links
            if title.strip() and url.strip()
        ]
        self._timeout_seconds = max(timeout_minutes * 60, 0)
        self._interval = max(interval_seconds, 1)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._logger = logging.getLogger("loop_bot.social_ad")
        self._tracking_url_getter = tracking_url_getter

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
        await self._disable_previous_more_button(user_id)
        if not self._social_links:
            self._logger.warning("No social links configured; skipping social ad")
            return
        # Get tracking URL if available
        tracking_url = None
        if self._tracking_url_getter:
            try:
                tracking_url = self._tracking_url_getter()
            except Exception:
                pass
        keyboard = social_ad_keyboard(
            self._social_links,
            tracking_base_url=tracking_url,
            user_id=user_id,
        )
        text = f"<b>{msg.SOCIAL_AD_TITLE}</b>\n{msg.SOCIAL_AD_BODY}"
        message = None
        try:
            message = await self._bot.send_message(
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
            if message is not None:
                try:
                    await track_event(str(user_id), "social_ad_shown")
                except Exception:  # pragma: no cover - analytics not initialized
                    self._logger.debug("Analytics not available for social_ad_shown")
                await self._repository.set_last_more_message(
                    user_id,
                    message.message_id,
                    "social",
                    {
                        "links": [
                            {"title": title, "url": url}
                            for title, url in self._social_links
                        ]
                    },
                )

    async def _disable_previous_more_button(self, user_id: int) -> None:
        profile = await self._repository.ensure_user(user_id)
        message_id = profile.last_more_message_id
        message_type = profile.last_more_message_type
        if not message_id or not message_type:
            return
        markup = more_buttonless_markup(
            message_type, profile.last_more_message_payload
        )
        if markup is None:
            await self._repository.set_last_more_message(user_id, None, None, None)
            return
        try:
            await self._bot.edit_message_reply_markup(
                chat_id=user_id,
                message_id=message_id,
                reply_markup=markup,
            )
        except (TelegramForbiddenError, TelegramBadRequest) as exc:
            self._logger.debug(
                "Failed to update previous more button for %s: %s", user_id, exc
            )
        finally:
            await self._repository.set_last_more_message(user_id, None, None, None)
