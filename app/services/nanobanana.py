"""Client for the NanoBanana (Google Gemini) image try-on API."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
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

UNSUITABLE_CODE = "UNSUITABLE_PHOTO"
TRANSIENT_CODE = "TRANSIENT"
PARSER_MISS_CODE = "PARSER_MISS"

_API_KEY: str | None = None
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
_RETRY_DELAYS = (1.0, 3.0)


@dataclass(slots=True)
class GenerationSuccess:
    """Successful generation payload with metadata."""

    image_bytes: bytes
    response: dict[str, Any]
    finish_reason: str | None
    has_inline: bool
    has_data_url: bool
    has_file_uri: bool
    attempt: int
    retried: bool


@dataclass(slots=True)
class _ResponseScan:
    inline_data: str | None
    finish_reason: str | None
    has_inline: bool
    has_data_url: bool
    has_file_uri: bool
    has_safety: bool


class NanoBananaGenerationError(RuntimeError):
    """Error raised when Gemini fails to return a usable image."""

    def __init__(
        self,
        message: str,
        *,
        response: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        reason_code: str = TRANSIENT_CODE,
        reason_detail: str = "",
        has_inline: bool = False,
        has_data_url: bool = False,
        has_file_uri: bool = False,
        attempt: int = 1,
        retried: bool = False,
    ) -> None:
        super().__init__(message)
        self.response = response
        self.finish_reason = finish_reason
        self.reason_code = reason_code
        self.reason_detail = reason_detail
        self.has_inline = has_inline
        self.has_data_url = has_data_url
        self.has_file_uri = has_file_uri
        self.attempt = attempt
        self.retried = retried

    def with_attempt(self, attempt: int, retried: bool) -> "NanoBananaGenerationError":
        self.attempt = attempt
        self.retried = retried
        return self


def configure(api_key: str) -> None:
    """Configure the API key used for subsequent requests."""

    global _API_KEY
    sanitized = (api_key or "").strip()
    if not sanitized:
        raise RuntimeError("NANOBANANA_API_KEY is not configured")
    _API_KEY = sanitized


async def generate_glasses(face_path: str, glasses_path: str) -> GenerationSuccess:
    """Return generated data with metadata from Gemini; raise on failure."""

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

    url = f"{API_ENDPOINT}?key={_API_KEY}"
    attempt = 0
    while True:
        attempt += 1
        retried = attempt > 1
        try:
            data = await _request_generation(url, payload)
        except NanoBananaGenerationError as exc:
            exc = exc.with_attempt(attempt, retried)
            if (
                exc.reason_code == TRANSIENT_CODE
                and attempt <= len(_RETRY_DELAYS)
            ):
                await asyncio.sleep(_RETRY_DELAYS[attempt - 1])
                continue
            raise exc

        scan = _scan_response(data)
        if scan.inline_data:
            try:
                decoded = base64.b64decode(scan.inline_data, validate=True)
            except (ValueError, TypeError) as exc:  # pragma: no cover - unexpected format
                reason_code, reason_detail = classify_failure(
                    data, exc, scan=scan
                )
                error = NanoBananaGenerationError(
                    "Invalid base64 image in Gemini response",
                    response=data,
                    finish_reason=scan.finish_reason,
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    has_inline=scan.has_inline,
                    has_data_url=scan.has_data_url,
                    has_file_uri=scan.has_file_uri,
                ).with_attempt(attempt, retried)
                if (
                    error.reason_code == TRANSIENT_CODE
                    and attempt <= len(_RETRY_DELAYS)
                ):
                    await asyncio.sleep(_RETRY_DELAYS[attempt - 1])
                    continue
                raise error from exc
            return GenerationSuccess(
                image_bytes=decoded,
                response=data,
                finish_reason=scan.finish_reason,
                has_inline=scan.has_inline,
                has_data_url=scan.has_data_url,
                has_file_uri=scan.has_file_uri,
                attempt=attempt,
                retried=retried,
            )

        reason_code, reason_detail = classify_failure(data, scan=scan)
        error = NanoBananaGenerationError(
            "Gemini response missing image data",
            response=data,
            finish_reason=scan.finish_reason,
            reason_code=reason_code,
            reason_detail=reason_detail,
            has_inline=scan.has_inline,
            has_data_url=scan.has_data_url,
            has_file_uri=scan.has_file_uri,
        ).with_attempt(attempt, retried)
        if error.reason_code == TRANSIENT_CODE and attempt <= len(_RETRY_DELAYS):
            await asyncio.sleep(_RETRY_DELAYS[attempt - 1])
            continue
        raise error


def classify_failure(
    response: dict[str, Any] | None,
    error: Exception | None = None,
    *,
    status: int | None = None,
    scan: _ResponseScan | None = None,
) -> tuple[str, str]:
    """Classify the reason for a failed generation attempt."""

    if response is not None:
        if scan is None:
            scan = _scan_response(response)
        finish_reason_raw = scan.finish_reason or ""
        finish_reason = finish_reason_raw.upper()
        has_candidates = _has_candidates(response)
        inline_present = scan.inline_data is not None
        has_alternative_refs = scan.has_data_url or scan.has_file_uri

        safety_hit = False
        safety_detail = ""
        if not inline_present:
            safety_hit, safety_detail = _detect_safety_signal(response)

        if safety_hit:
            return UNSUITABLE_CODE, safety_detail

        if not inline_present:
            if finish_reason in {"SAFETY", "BLOCKED"}:
                detail_parts: list[str] = []
                if finish_reason:
                    detail_parts.append(f"finish={finish_reason}")
                if not has_candidates:
                    detail_parts.append("empty_candidates")
                if not detail_parts:
                    detail_parts.append("finish=UNKNOWN")
                return UNSUITABLE_CODE, ",".join(detail_parts)

            if finish_reason in {"OTHER", "IMAGE_OTHER", "ERROR"}:
                detail_parts = []
                if finish_reason:
                    detail_parts.append(f"finish={finish_reason}")
                if not has_candidates:
                    detail_parts.append("empty_candidates")
                if not has_alternative_refs:
                    detail_parts.append("no_parts")
                if not detail_parts:
                    detail_parts.append("finish=UNKNOWN")
                return TRANSIENT_CODE, ",".join(detail_parts)

            if has_alternative_refs:
                detail_parts = []
                if finish_reason:
                    detail_parts.append(f"finish={finish_reason}")
                detail_parts.append("alt_refs")
                return PARSER_MISS_CODE, ",".join(detail_parts)

            if not has_candidates:
                return TRANSIENT_CODE, "empty_candidates"

            return PARSER_MISS_CODE, "no_image_data"

        if finish_reason in {"SAFETY", "BLOCKED"}:
            detail = f"finish={finish_reason}" if finish_reason else "finish=UNKNOWN"
            return UNSUITABLE_CODE, detail

        if not has_candidates:
            return TRANSIENT_CODE, "empty_candidates"

        return PARSER_MISS_CODE, "no_image_data"

    if error is not None:
        if isinstance(error, asyncio.TimeoutError):
            return TRANSIENT_CODE, "timeout"
        if isinstance(error, aiohttp.ClientResponseError):
            status = error.status
            if status >= 500 or status == 429:
                return TRANSIENT_CODE, f"status={status}"
            return PARSER_MISS_CODE, f"status={status}"
        if isinstance(error, aiohttp.ClientError):
            return TRANSIENT_CODE, error.__class__.__name__
        if isinstance(error, json.JSONDecodeError):
            return PARSER_MISS_CODE, "json_decode"
        return TRANSIENT_CODE, error.__class__.__name__

    if status is not None:
        if status >= 500 or status == 429:
            return TRANSIENT_CODE, f"status={status}"
        return PARSER_MISS_CODE, f"status={status}"

    return PARSER_MISS_CODE, "unknown"


def _detect_safety_signal(response: dict[str, Any]) -> tuple[bool, str]:
    """Inspect safety ratings and detect blocked or high-risk content."""

    ratings = _collect_safety_ratings(response)
    for rating in ratings:
        if not isinstance(rating, dict):
            continue
        category_value = rating.get("category") or rating.get("name") or ""
        matched, category_token = _match_safety_category(str(category_value))
        if not matched:
            continue
        if _is_blocked_value(rating.get("blocked")):
            return True, f"safety={category_token}/blocked"
        level = _extract_level_token(rating)
        if level:
            return True, f"safety={category_token}/{level}"

    return False, ""


def _collect_safety_ratings(payload: Any) -> list[Any]:
    """Collect safety rating dictionaries from a nested payload."""

    results: list[Any] = []
    queue: list[Any] = [payload]
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in {"safetyRatings", "safety_ratings"} and isinstance(value, list):
                    results.extend(value)
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)
    return results


_SAFETY_KEYWORDS = (
    "image_violence",
    "violence",
    "harassment",
    "hate",
    "sexual",
    "child_safety",
    "dangerous",
    "medical",
)


def _match_safety_category(category: str) -> tuple[bool, str]:
    normalized = category.replace("-", "_").lower()
    for keyword in _SAFETY_KEYWORDS:
        if keyword in normalized:
            return True, keyword
    return False, normalized or "unknown"


def _is_blocked_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"true", "blocked", "1", "yes"}
    return False


def _extract_level_token(rating: dict[str, Any]) -> str:
    """Extract a normalized severity level token from a safety rating."""

    for key in ("probability", "probabilityScore", "probability_score", "likelihood", "severity"):
        if key not in rating:
            continue
        level_raw = _normalize_level(rating[key])
        if not level_raw:
            continue
        token = _level_token(level_raw)
        if token:
            return token
    return ""


def _normalize_level(value: Any) -> str:
    if isinstance(value, str):
        upper = value.strip().upper()
        for prefix in (
            "PROBABILITY_",
            "LIKELIHOOD_",
            "HARM_CATEGORY_",
            "CATEGORY_",
            "LEVEL_",
            "SAFETY_",
            "BLOCKED_",
        ):
            if upper.startswith(prefix):
                upper = upper[len(prefix) :]
        if upper.startswith("SCORE_"):
            upper = upper[len("SCORE_") :]
        if "AND_ABOVE" in upper:
            upper = upper.replace("AND_ABOVE", "")
        upper = upper.strip("_ ")
        return upper
    if isinstance(value, (int, float)):
        if value >= 0.75:
            return "HIGH"
        if value >= 0.5:
            return "MEDIUM"
        return ""
    if isinstance(value, dict):
        for key in ("label", "value", "name"):
            if key in value:
                normalized = _normalize_level(value[key])
                if normalized:
                    return normalized
    return ""


def _level_token(level: str) -> str:
    upper = level.upper()
    if "HIGH" in upper:
        return "high"
    if "MEDIUM" in upper:
        return "medium"
    return ""


async def _request_generation(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
        try:
            async with session.post(url, json=payload) as response:
                text = await response.text()
                if response.status != 200:
                    reason_code, reason_detail = classify_failure(
                        None, status=response.status
                    )
                    raise NanoBananaGenerationError(
                        f"Gemini API returned status {response.status}",
                        reason_code=reason_code,
                        reason_detail=f"{reason_detail}:{text[:120]}",
                    )
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    reason_code, reason_detail = classify_failure(None, exc)
                    raise NanoBananaGenerationError(
                        "Failed to decode Gemini response as JSON",
                        reason_code=reason_code,
                        reason_detail=reason_detail,
                    ) from exc
        except asyncio.TimeoutError as exc:  # pragma: no cover - network failures
            reason_code, reason_detail = classify_failure(None, exc)
            raise NanoBananaGenerationError(
                "Gemini request timed out",
                reason_code=reason_code,
                reason_detail=reason_detail,
            ) from exc
        except aiohttp.ClientError as exc:  # pragma: no cover - network failures
            reason_code, reason_detail = classify_failure(None, exc)
            raise NanoBananaGenerationError(
                "Gemini request error",
                reason_code=reason_code,
                reason_detail=reason_detail,
            ) from exc


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


def _scan_response(payload: Any) -> _ResponseScan:
    inline_data: str | None = None
    finish_reason: str | None = None
    has_inline = False
    has_data_url = False
    has_file_uri = False
    has_safety = False

    queue: list[Any] = [payload]
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in {"inline_data", "inlineData"} and isinstance(value, dict):
                    has_inline = True
                    data_value = value.get("data")
                    if isinstance(data_value, str) and data_value.strip():
                        inline_data = data_value
                elif key in {"finishReason", "finish_reason"} and isinstance(value, str):
                    if not finish_reason:
                        finish_reason = value
                elif key in {"fileUri", "file_uri"} and isinstance(value, str):
                    has_file_uri = True
                elif key in {"safetyRatings", "safety_ratings"} and isinstance(value, list) and value:
                    has_safety = True
                elif key.lower() in {"data", "uri", "url"} and isinstance(value, str):
                    if value.startswith("data:"):
                        has_data_url = True
                    if key.lower() in {"uri", "url"} and (
                        value.startswith("gs://")
                        or "googleapis" in value
                    ):
                        has_file_uri = True
                elif isinstance(value, str):
                    if value.startswith("data:"):
                        has_data_url = True
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)
        elif isinstance(current, str):
            if current.startswith("data:"):
                has_data_url = True

    return _ResponseScan(
        inline_data=inline_data,
        finish_reason=finish_reason,
        has_inline=has_inline,
        has_data_url=has_data_url,
        has_file_uri=has_file_uri,
        has_safety=has_safety,
    )


def _has_candidates(response: dict[str, Any]) -> bool:
    candidates = response.get("candidates") if isinstance(response, dict) else None
    return bool(candidates)


__all__ = [
    "GenerationSuccess",
    "NanoBananaGenerationError",
    "classify_failure",
    "configure",
    "generate_glasses",
]

