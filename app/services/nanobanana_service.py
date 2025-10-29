"""High-level helper wrapping the NanoBanana client."""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.services.nanobanana_client import NanoBananaClient, NanoBananaError

LOGGER = logging.getLogger(__name__)
_FALLBACK_NOTICE = "генерация временно недоступна, показан анализ"

_CLIENT: Optional[NanoBananaClient] = None


def _get_client() -> NanoBananaClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = NanoBananaClient()
    return _CLIENT


def _get_mode() -> str:
    mode = os.getenv("NANOBANANA_MODE", "analyze").strip().lower()
    if mode not in {"analyze", "image"}:
        return "analyze"
    return mode


def _should_fallback(error: NanoBananaError) -> bool:
    if error.status_code in {403, 429}:
        return True
    return error.is_quota_related


def generate_or_describe(face_bytes: bytes, glasses_bytes: Optional[bytes] = None) -> dict[str, object]:
    """Generate a try-on image or describe the source depending on the mode."""

    client = _get_client()
    mode = _get_mode()

    if mode != "image":
        try:
            description = client.analyze_image(face_bytes)
        except Exception:  # pragma: no cover - network failure
            LOGGER.error("NanoBanana analyze request failed", exc_info=True)
            raise
        LOGGER.info("NanoBanana analyze request succeeded")
        return {"mode": "analyze", "text": description}

    if glasses_bytes is None:
        raise ValueError("glasses_bytes is required when NANOBANANA_MODE=image")

    try:
        generated = client.put_glasses(face_bytes, glasses_bytes)
    except NanoBananaError as error:
        if _should_fallback(error):
            LOGGER.warning(
                "NanoBanana image generation unavailable (status=%s): %s",
                error.status_code,
                error,
            )
            try:
                description = client.analyze_image(face_bytes)
            except Exception:  # pragma: no cover - secondary failure
                LOGGER.error(
                    "NanoBanana fallback analyze failed after generation error",
                    exc_info=True,
                )
                raise
            text = f"{_FALLBACK_NOTICE}: {description}"
            return {"mode": "analyze", "text": text}
        LOGGER.error("NanoBanana image generation failed", exc_info=True)
        raise
    except Exception:
        LOGGER.error("Unexpected NanoBanana image generation error", exc_info=True)
        raise

    LOGGER.info("NanoBanana image generation succeeded")
    return {"mode": "image", "image_bytes": generated}
