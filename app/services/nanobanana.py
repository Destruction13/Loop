"""Client for the NanoBanana (Google Gemini) image try-on API."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from logger import get_logger


LOGGER = get_logger("generation.nanobanana")

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


@dataclass(slots=True)
class SafetySummary:
    """Structured representation of detected safety signals."""

    present: bool
    triggered: bool
    detail: str
    categories: tuple[str, ...]
    levels: dict[str, str]

    @classmethod
    def empty(cls) -> "SafetySummary":
        return cls(
            present=False,
            triggered=False,
            detail="",
            categories=(),
            levels={},
        )


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
        safety_present: bool = False,
        safety_categories: tuple[str, ...] | list[str] | None = None,
        safety_levels: dict[str, str] | None = None,
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
        self.safety_present = safety_present
        self.safety_categories = tuple(safety_categories or ())
        self.safety_levels = dict(safety_levels or {})

def configure(api_key: str) -> None:
    """Configure the API key used for subsequent requests."""

    global _API_KEY
    sanitized = (api_key or "").strip()
    if not sanitized:
        raise RuntimeError("NANOBANANA_API_KEY is not configured")
    _API_KEY = sanitized
    LOGGER.info("API NanoBanana сконфигурирован", extra={"stage": "NANO_CONFIG"})


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
    data = await _request_generation(url, payload)

    scan = _scan_response(data)
    if scan.inline_data:
        try:
            decoded = base64.b64decode(scan.inline_data, validate=True)
        except (ValueError, TypeError) as exc:  # pragma: no cover - unexpected format
            reason_code, reason_detail, safety = classify_failure(data, exc, scan=scan)
            raise NanoBananaGenerationError(
                "Invalid base64 image in Gemini response",
                response=data,
                finish_reason=scan.finish_reason,
                reason_code=reason_code,
                reason_detail=reason_detail,
                has_inline=scan.has_inline,
                has_data_url=scan.has_data_url,
                has_file_uri=scan.has_file_uri,
                safety_present=safety.present,
                safety_categories=safety.categories,
                safety_levels=safety.levels,
            ) from exc
        return GenerationSuccess(
            image_bytes=decoded,
            response=data,
            finish_reason=scan.finish_reason,
            has_inline=scan.has_inline,
            has_data_url=scan.has_data_url,
            has_file_uri=scan.has_file_uri,
            attempt=1,
            retried=False,
        )

    reason_code, reason_detail, safety = classify_failure(data, scan=scan)
    raise NanoBananaGenerationError(
        "Gemini response missing image data",
        response=data,
        finish_reason=scan.finish_reason,
        reason_code=reason_code,
        reason_detail=reason_detail,
        has_inline=scan.has_inline,
        has_data_url=scan.has_data_url,
        has_file_uri=scan.has_file_uri,
        safety_present=safety.present,
        safety_categories=safety.categories,
        safety_levels=safety.levels,
    )


def classify_failure(
    response: dict[str, Any] | None,
    error: Exception | None = None,
    *,
    status: int | None = None,
    scan: _ResponseScan | None = None,
) -> tuple[str, str, SafetySummary]:
    """Classify the reason for a failed generation attempt."""

    safety_summary = SafetySummary.empty()

    if response is not None:
        if scan is None:
            scan = _scan_response(response)
        finish_reason_raw = scan.finish_reason or ""
        finish_reason = finish_reason_raw.upper()
        has_candidates = _has_candidates(response)
        inline_present = scan.inline_data is not None
        has_alternative_refs = scan.has_data_url or scan.has_file_uri

        safety_summary = _analyse_safety_signals(response)

        detail_parts: list[str] = []
        safety_detail = safety_summary.detail
        if safety_detail:
            detail_parts.append(safety_detail)
        elif safety_summary.present and safety_summary.triggered:
            detail_parts.append("safety_present")
        if not detail_parts and finish_reason:
            detail_parts.append(f"finish={finish_reason}")
        if not has_candidates and not safety_detail:
            detail_parts.append("empty_candidates")
        if not has_alternative_refs and not safety_detail:
            detail_parts.append("no_parts")
        if has_alternative_refs and not inline_present:
            detail_parts.append("alt_refs")

        if not inline_present:
            if not detail_parts:
                detail_parts.append("no_image_data")
            return UNSUITABLE_CODE, ",".join(detail_parts), safety_summary

        if error is not None:
            if not detail_parts:
                detail_parts.append("invalid_image_data")
            return UNSUITABLE_CODE, ",".join(detail_parts), safety_summary

        if not detail_parts:
            detail_parts.append("no_image_data")
        return UNSUITABLE_CODE, ",".join(detail_parts), safety_summary

    if error is not None:
        if isinstance(error, asyncio.TimeoutError):
            return TRANSIENT_CODE, "timeout", safety_summary
        if isinstance(error, aiohttp.ClientResponseError):
            status = error.status
            if status >= 500 or status == 429:
                return TRANSIENT_CODE, f"status={status}", safety_summary
            return PARSER_MISS_CODE, f"status={status}", safety_summary
        if isinstance(error, aiohttp.ClientError):
            return TRANSIENT_CODE, error.__class__.__name__, safety_summary
        if isinstance(error, json.JSONDecodeError):
            return PARSER_MISS_CODE, "json_decode", safety_summary
        return TRANSIENT_CODE, error.__class__.__name__, safety_summary

    if status is not None:
        if status >= 500 or status == 429:
            return TRANSIENT_CODE, f"status={status}", safety_summary
        return PARSER_MISS_CODE, f"status={status}", safety_summary

    return PARSER_MISS_CODE, "unknown", safety_summary


def _analyse_safety_signals(response: dict[str, Any]) -> SafetySummary:
    """Inspect safety ratings and aggregate their severities."""

    ratings = _collect_safety_ratings(response)
    categories_in_order: list[str] = []
    levels: dict[str, str] = {}
    triggered = False
    detail = ""
    detail_priority = -1

    for rating in ratings:
        if not isinstance(rating, dict):
            continue
        category_value = rating.get("category") or rating.get("name") or ""
        matched, category_token = _match_safety_category(str(category_value))
        if not matched:
            continue
        if category_token not in categories_in_order:
            categories_in_order.append(category_token)

        blocked = _is_blocked_value(rating.get("blocked"))
        level_token = _extract_level_token(rating)
        if blocked:
            _merge_level(levels, category_token, "BLOCKED")
        elif level_token:
            _merge_level(levels, category_token, level_token.upper())

        if blocked:
            triggered = True
            blocked_priority = _LEVEL_PRIORITY.get("BLOCKED", 4)
            if detail_priority < blocked_priority:
                detail = f"safety={category_token}/blocked"
                detail_priority = blocked_priority
            continue

        if level_token:
            severity_priority = _LEVEL_PRIORITY.get(level_token.upper(), 0)
            if severity_priority >= _LEVEL_PRIORITY.get("MEDIUM", 2):
                triggered = True
                if severity_priority > detail_priority:
                    detail = f"safety={category_token}/{level_token}"
                    detail_priority = severity_priority

    present = bool(categories_in_order)
    ordered_levels = {cat: levels[cat] for cat in categories_in_order if cat in levels}
    return SafetySummary(
        present=present,
        triggered=triggered,
        detail=detail,
        categories=tuple(categories_in_order),
        levels=ordered_levels,
    )


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
    "graphic",
    "blood",
    "harassment",
    "threat",
    "hate",
    "sexual",
    "child_safety",
    "self_harm",
    "suicide",
    "dangerous",
    "weapons",
    "firearm",
    "terrorism",
    "extremism",
    "medical",
    "drugs",
)


def _match_safety_category(category: str) -> tuple[bool, str]:
    normalized = category.replace("-", "_").lower().strip()
    for prefix in ("harm_category_", "harm_", "category_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
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

    for key in (
        "probability",
        "probabilityScore",
        "probability_score",
        "prob",
        "probabilityLevel",
        "likelihood",
        "likelihoodLevel",
        "severity",
        "severityLevel",
        "confidence",
    ):
        if key not in rating:
            continue
        level_raw = _normalize_level(rating[key])
        if not level_raw:
            continue
        token = _level_token(level_raw)
        if token:
            return token
    return ""


_LEVEL_PRIORITY = {
    "BLOCKED": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "VERY_LOW": 0,
}


def _merge_level(levels: dict[str, str], category: str, level: str) -> None:
    current_priority = _LEVEL_PRIORITY.get(level.upper(), 0)
    previous = levels.get(category)
    if previous is None or current_priority > _LEVEL_PRIORITY.get(previous, 0):
        levels[category] = level.upper()


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
    if "LOW" in upper:
        if "VERY" in upper:
            return "very_low"
        return "low"
    return ""


async def _request_generation(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
        try:
            async with session.post(url, json=payload) as response:
                text = await response.text()
                if response.status != 200:
                    reason_code, reason_detail, safety = classify_failure(
                        None, status=response.status
                    )
                    raise NanoBananaGenerationError(
                        f"Gemini API returned status {response.status}",
                        reason_code=reason_code,
                        reason_detail=f"{reason_detail}:{text[:120]}",
                        safety_present=safety.present,
                        safety_categories=safety.categories,
                        safety_levels=safety.levels,
                    )
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    reason_code, reason_detail, safety = classify_failure(None, exc)
                    raise NanoBananaGenerationError(
                        "Failed to decode Gemini response as JSON",
                        reason_code=reason_code,
                        reason_detail=reason_detail,
                        safety_present=safety.present,
                        safety_categories=safety.categories,
                        safety_levels=safety.levels,
                    ) from exc
        except asyncio.TimeoutError as exc:  # pragma: no cover - network failures
            reason_code, reason_detail, safety = classify_failure(None, exc)
            raise NanoBananaGenerationError(
                "Gemini request timed out",
                reason_code=reason_code,
                reason_detail=reason_detail,
                safety_present=safety.present,
                safety_categories=safety.categories,
                safety_levels=safety.levels,
            ) from exc
        except aiohttp.ClientError as exc:  # pragma: no cover - network failures
            reason_code, reason_detail, safety = classify_failure(None, exc)
            raise NanoBananaGenerationError(
                "Gemini request error",
                reason_code=reason_code,
                reason_detail=reason_detail,
                safety_present=safety.present,
                safety_categories=safety.categories,
                safety_levels=safety.levels,
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
    "SafetySummary",
    "classify_failure",
    "configure",
    "generate_glasses",
]

