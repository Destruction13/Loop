"""Deep link helpers."""

from __future__ import annotations

from urllib.parse import quote_plus


def build_ref_link(bot_username: str, user_id: int) -> str:
    """Construct referral deep link."""

    return f"https://t.me/{bot_username}?start=ref_{quote_plus(str(user_id))}"
