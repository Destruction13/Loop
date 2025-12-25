"""Admin access control helpers."""

from __future__ import annotations

from typing import Iterable

# Replace with real Telegram user IDs for admin access.
ADMIN_WHITELIST: set[int] = {123456789}


def is_admin(user_id: int | None, *, whitelist: Iterable[int] | None = None) -> bool:
    """Return True when the user is allowed to access admin features."""

    if user_id is None:
        return False
    allowed = set(whitelist) if whitelist is not None else ADMIN_WHITELIST
    return int(user_id) in allowed
