from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware

from app.services.repository import Repository


class UserIdentityMiddleware(BaseMiddleware):
    """Persist basic Telegram identity data for admin reporting."""

    def __init__(self, repository: Repository) -> None:
        self._repository = repository

    async def __call__(
        self,
        handler: Callable[[Any, dict[str, Any]], Awaitable[Any]],
        event: Any,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is not None:
            user_id = getattr(user, "id", None)
            if user_id is not None:
                username = getattr(user, "username", None)
                first_name = getattr(user, "first_name", None)
                last_name = getattr(user, "last_name", None)
                full_name = " ".join(
                    part for part in (first_name, last_name) if part
                ).strip()
                try:
                    await self._repository.upsert_user_identity(
                        int(user_id),
                        username=username,
                        first_name=first_name,
                        last_name=last_name,
                        full_name=full_name or None,
                    )
                except Exception:
                    # Identity capture is best-effort and should not block the bot.
                    pass
        return await handler(event, data)
