"""Telegram WebApp initData verification."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl


@dataclass(frozen=True)
class VerifiedInitData:
    """Verified initData payload with parsed user data."""

    user_id: int
    user: dict[str, Any]
    auth_date: int | None
    raw: dict[str, str]


def _build_data_check_string(items: dict[str, str]) -> str:
    return "\n".join(f"{key}={items[key]}" for key in sorted(items.keys()))


def _calc_hash(data_check_string: str, bot_token: str) -> str:
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_init_data(init_data: str, bot_token: str) -> VerifiedInitData | None:
    """Validate initData signature and return parsed payload on success."""

    if not init_data or not bot_token:
        return None
    parsed_pairs = parse_qsl(init_data, keep_blank_values=True)
    payload: dict[str, str] = {key: value for key, value in parsed_pairs}
    received_hash = payload.pop("hash", None)
    if not received_hash:
        return None
    data_check_string = _build_data_check_string(payload)
    calculated_hash = _calc_hash(data_check_string, bot_token)
    if not hmac.compare_digest(received_hash, calculated_hash):
        return None

    user_raw = payload.get("user")
    if not user_raw:
        return None
    try:
        user = json.loads(user_raw)
    except json.JSONDecodeError:
        return None
    try:
        user_id = int(user.get("id"))
    except (TypeError, ValueError):
        return None

    auth_date = None
    if "auth_date" in payload:
        try:
            auth_date = int(payload["auth_date"])
        except (TypeError, ValueError):
            auth_date = None
    return VerifiedInitData(user_id=user_id, user=user, auth_date=auth_date, raw=payload)
