from __future__ import annotations

import uuid
from typing import Any, Callable, Awaitable

from aiogram import BaseMiddleware

from logger import bind_context, reset_context


class LoggingMiddleware(BaseMiddleware):
    """Inject request_id and user_id into logging context for each update."""

    async def __call__(
        self,
        handler: Callable[[Any, dict[str, Any]], Awaitable[Any]],
        event: Any,
        data: dict[str, Any],
    ) -> Any:
        request_id = uuid.uuid4().hex[:8]
        user = data.get("event_from_user")
        user_id = getattr(user, "id", None)

        data["request_id"] = request_id
        if user_id is not None:
            data["user_id"] = user_id

        tokens = bind_context(request_id=request_id, user_id=user_id)
        try:
            return await handler(event, data)
        finally:
            reset_context(tokens)
