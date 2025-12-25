import hashlib
import hmac
import json
from urllib.parse import urlencode

from app.admin.init_data import verify_init_data
from app.admin.security import is_admin


def _build_init_data(bot_token: str, *, user_id: int) -> str:
    payload = {
        "auth_date": "1700000000",
        "query_id": "AAE2b7c1234567890",
        "user": json.dumps({"id": user_id, "username": "tester"}, separators=(",", ":")),
    }
    data_check_string = "\n".join(
        f"{key}={payload[key]}" for key in sorted(payload.keys())
    )
    secret_key = hmac.new(
        b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256
    ).digest()
    signature = hmac.new(
        secret_key, data_check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    payload["hash"] = signature
    return urlencode(payload)


def test_verify_init_data_valid() -> None:
    init_data = _build_init_data("test-token", user_id=42)
    verified = verify_init_data(init_data, "test-token")
    assert verified is not None
    assert verified.user_id == 42
    assert verified.user["username"] == "tester"


def test_verify_init_data_invalid() -> None:
    init_data = _build_init_data("test-token", user_id=42)
    bad = init_data.replace("hash=", "hash=deadbeef")
    assert verify_init_data(bad, "test-token") is None


def test_admin_whitelist() -> None:
    assert is_admin(123456789, whitelist={123456789})
    assert not is_admin(123, whitelist={123456789})
