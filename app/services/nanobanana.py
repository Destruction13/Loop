"""Client for the NanoBanana (Google Gemini) image try-on API."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import aiohttp

SYSTEM_PROMPT = (
    "Take the first image as the base photo (it can be a human, animal, or any face).\n"
    "Take the second image as the glasses reference.\n"
    "Place the glasses from the second image naturally and convincingly on the face in the first image.\n"
    "Make sure the glasses fit perfectly — keep correct proportions, perspective, and realistic positioning on the nose and ears (if visible).\n"
    "Do not distort or change the original face or its features, only add the glasses.\n"
    "Keep natural lighting, shadows, and reflections consistent with the base image.\n"
    "The final result should look like a real photo where the glasses are actually worn."
)

MODEL_NAME = "gemini-2.5-flash-image"
API_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)

_API_KEY: str | None = None
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
_RETRY_DELAYS = (1.0, 3.0)


def configure(api_key: str) -> None:
    """Configure the API key used for subsequent requests."""

    global _API_KEY
    sanitized = (api_key or "").strip()
    if not sanitized:
        raise RuntimeError("NANOBANANA_API_KEY is not configured")
    _API_KEY = sanitized


async def generate_glasses(face_path: str, glasses_path: str) -> bytes:
    """Return generated PNG bytes from Gemini; raise RuntimeError on failure."""

    if not _API_KEY:
        raise RuntimeError("NanoBanana API key is not set")
    face_payload = _encode_image(Path(face_path))
    glasses_payload = _encode_image(Path(glasses_path))
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [face_payload, glasses_payload],
            }
        ],
    }

    attempt = 0
    last_error: Exception | None = None
    url = f"{API_ENDPOINT}?key={_API_KEY}"
    while attempt < 1 + len(_RETRY_DELAYS):
        attempt += 1
        try:
            return await _request_generation(url, payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt > len(_RETRY_DELAYS):
                break
            await asyncio.sleep(_RETRY_DELAYS[attempt - 1])
    message = "Gemini generation failed"
    if last_error:
        message = f"{message}: {last_error}"
    raise RuntimeError(message)


async def _request_generation(url: str, payload: dict[str, Any]) -> bytes:
    async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(
                        f"Gemini API returned status {response.status}: {text[:200]}"
                    )
                data = await response.json()
        except aiohttp.ClientError as exc:  # pragma: no cover - network failures
            raise RuntimeError(f"Gemini request error: {exc}") from exc
        except asyncio.TimeoutError as exc:  # pragma: no cover - network failures
            raise RuntimeError("Gemini request timed out") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to decode Gemini response as JSON") from exc
    encoded = _extract_inline_data(data)
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Invalid base64 image in Gemini response") from exc


def _encode_image(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "inline_data": {
            "mime_type": _guess_mime(path),
            "data": encoded,
        }
    }


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "image/png"


def _extract_inline_data(payload: Any) -> str:
    queue: list[Any] = [payload]
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            for key in ("inline_data", "inlineData"):
                if key in current and isinstance(current[key], dict):
                    data_value = current[key].get("data")
                    if isinstance(data_value, str) and data_value.strip():
                        return data_value
            for value in current.values():
                queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)
    raise RuntimeError("Gemini response does not contain inline image data")


__all__ = ["configure", "generate_glasses"]

