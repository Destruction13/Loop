"""Client for interacting with Google Gemini (NanoBanana) API."""

from __future__ import annotations

import base64
import binascii
import imghdr
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


def _guess_mime_type(data: bytes) -> str:
    """Best-effort detection of the image MIME type."""

    kind = imghdr.what(None, data)
    if kind == "png":
        return "image/png"
    if kind in {"jpeg", "jpg"}:
        return "image/jpeg"
    if kind == "gif":
        return "image/gif"
    if kind == "webp":
        return "image/webp"
    return "image/jpeg"


def _b64(image_bytes: bytes) -> str:
    """Encode raw bytes to a base64 string suitable for JSON payloads."""

    return base64.b64encode(image_bytes).decode("ascii")


@dataclass(slots=True)
class NanoBananaError(RuntimeError):
    """Error raised for HTTP or API level failures."""

    message: str
    status_code: Optional[int] = None
    payload: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        RuntimeError.__init__(self, self.message)

    @property
    def is_quota_related(self) -> bool:
        if not self.payload:
            return False
        try:
            serialized = json.dumps(self.payload)
        except (TypeError, ValueError):
            return False
        return "quota" in serialized.lower()


class NanoBananaClient:
    """HTTP client for NanoBanana (Google Gemini) requests."""

    _ANALYZE_PROMPT = (
        "Опиши изображение: кто или что на нём изображено и важные детали. "
        "Ответь на русском языке."
    )

    def __init__(self) -> None:
        load_dotenv()
        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required")

        endpoint_base = os.getenv(
            "NANOBANANA_ENDPOINT_BASE", "https://generativelanguage.googleapis.com"
        )
        self._endpoint_base = endpoint_base.rstrip("/")
        self._timeout_seconds = self._parse_timeout(os.getenv("NANOBANANA_TIMEOUT"))

        self._model_analyze = os.getenv("NANOBANANA_MODEL_ANALYZE", "gemini-2.5-flash")
        self._model_image = os.getenv(
            "NANOBANANA_MODEL_IMAGE", "gemini-2.5-flash-image"
        )

        timeout = httpx.Timeout(self._timeout_seconds, connect=self._timeout_seconds)
        self._client = httpx.Client(timeout=timeout)

        self._headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        self._models_base_url = f"{self._endpoint_base}/v1beta/models"

    @staticmethod
    def _parse_timeout(value: Optional[str]) -> float:
        try:
            parsed = int(value) if value is not None else 180
        except ValueError:
            parsed = 180
        return max(float(parsed), 1.0)

    def _generate_url(self, model: str) -> str:
        return f"{self._models_base_url}/{model}:generateContent"

    def _post_with_retries(self, url: str, payload: Mapping[str, Any]) -> httpx.Response:
        retries = 4
        backoff = 0.8
        last_response: Optional[httpx.Response] = None
        for attempt in range(retries):
            try:
                response = self._client.post(url, headers=self._headers, json=payload)
            except httpx.TimeoutException as exc:  # pragma: no cover - network failure
                if attempt == retries - 1:
                    raise NanoBananaError("Request timed out") from exc
                sleep_for = backoff * (2 ** attempt)
                time.sleep(sleep_for)
                continue

            if response.status_code == 429 and attempt < retries - 1:
                sleep_for = backoff * (2 ** attempt)
                LOGGER.warning(
                    "NanoBanana responded with 429, retrying in %.1fs (attempt %s/%s)",
                    sleep_for,
                    attempt + 1,
                    retries,
                )
                time.sleep(sleep_for)
                last_response = response
                continue

            last_response = response
            break

        if last_response is None:  # pragma: no cover - defensive
            raise NanoBananaError("No response received from NanoBanana")

        return last_response

    @staticmethod
    def _extract_json(response: httpx.Response) -> Mapping[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise NanoBananaError(
                "Failed to decode JSON response", status_code=response.status_code
            ) from exc

    def _handle_error(self, response: httpx.Response) -> None:
        payload = None
        try:
            payload = response.json()
        except json.JSONDecodeError:  # pragma: no cover - best effort
            payload = None

        if response.status_code in {400, 403, 404}:
            message = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
            message = message or f"NanoBanana request failed with {response.status_code}"
            raise NanoBananaError(
                message,
                status_code=response.status_code,
                payload=payload if isinstance(payload, Mapping) else None,
            )

        if response.status_code >= 400:
            raise NanoBananaError(
                f"NanoBanana unexpected status {response.status_code}",
                status_code=response.status_code,
                payload=payload if isinstance(payload, Mapping) else None,
            )

    def analyze_image(self, image_bytes: bytes) -> str:
        """Request descriptive analysis of an image."""

        mime_type = _guess_mime_type(image_bytes)
        inline = {
            "mime_type": mime_type,
            "mimeType": mime_type,
            "data": _b64(image_bytes),
        }
        parts: list[dict[str, Any]] = [
            {"text": self._ANALYZE_PROMPT},
            {"inline_data": inline, "inlineData": inline},
        ]
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }
        response = self._post_with_retries(self._generate_url(self._model_analyze), payload)
        self._handle_error(response)
        data = self._extract_json(response)

        candidates = data.get("candidates") if isinstance(data, Mapping) else None
        if not candidates:
            prompt_feedback = (
                data.get("promptFeedback") if isinstance(data, Mapping) else None
            )
            block_reason = (
                prompt_feedback.get("blockReason") if isinstance(prompt_feedback, Mapping) else None
            )
            if block_reason:
                raise NanoBananaError(
                    f"Safety filters triggered: {block_reason}",
                    status_code=response.status_code,
                    payload=data if isinstance(data, Mapping) else None,
                )
            raise NanoBananaError(
                "NanoBanana returned no candidates",
                status_code=response.status_code,
                payload=data if isinstance(data, Mapping) else None,
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            text = part.get("text")
            if text:
                return text.strip()

        raise NanoBananaError(
            "NanoBanana response did not contain text parts",
            status_code=response.status_code,
            payload=data if isinstance(data, Mapping) else None,
        )

    def put_glasses(self, face_bytes: bytes, glasses_bytes: bytes) -> bytes:
        """Generate an image with glasses placed on a face."""

        face_mime = _guess_mime_type(face_bytes)
        glasses_mime = _guess_mime_type(glasses_bytes)
        face_inline = {
            "mime_type": face_mime,
            "mimeType": face_mime,
            "data": _b64(face_bytes),
        }
        glasses_inline = {
            "mime_type": glasses_mime,
            "mimeType": glasses_mime,
            "data": _b64(glasses_bytes),
        }
        system_prompt = (
            "Take the first image as the base photo (it can be a human, animal, or any face).\n"
            "Take the second image as the glasses reference.\n"
            "Place the glasses from the second image naturally and convincingly on the face in the first image.\n"
            "Make sure the glasses fit perfectly — keep correct proportions, perspective, and realistic positioning on the nose and ears (if visible).\n"
            "Do not distort or change the original face or its features, only add the glasses.\n"
            "Keep natural lighting, shadows, and reflections consistent with the base image.\n"
            "The final result should look like a real photo where the glasses are actually worn."
        )
        parts: list[dict[str, Any]] = [
            {"text": system_prompt},
            {"inline_data": face_inline, "inlineData": face_inline},
            {"inline_data": glasses_inline, "inlineData": glasses_inline},
        ]
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }
        response = self._post_with_retries(self._generate_url(self._model_image), payload)
        self._handle_error(response)
        data = self._extract_json(response)

        candidates = data.get("candidates") if isinstance(data, Mapping) else None
        if not candidates:
            prompt_feedback = (
                data.get("promptFeedback") if isinstance(data, Mapping) else None
            )
            block_reason = (
                prompt_feedback.get("blockReason") if isinstance(prompt_feedback, Mapping) else None
            )
            if block_reason:
                raise NanoBananaError(
                    f"Safety filters triggered: {block_reason}",
                    status_code=response.status_code,
                    payload=data if isinstance(data, Mapping) else None,
                )
            raise NanoBananaError(
                "NanoBanana returned no candidates",
                status_code=response.status_code,
                payload=data if isinstance(data, Mapping) else None,
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            inline_data = part.get("inline_data") or part.get("inlineData")
            if inline_data and inline_data.get("data"):
                try:
                    return base64.b64decode(inline_data["data"], validate=True)
                except (KeyError, binascii.Error, ValueError) as exc:
                    raise NanoBananaError(
                        "Failed to decode image data from NanoBanana",
                        status_code=response.status_code,
                        payload=data if isinstance(data, Mapping) else None,
                    ) from exc

        raise NanoBananaError(
            "NanoBanana response did not contain image data",
            status_code=response.status_code,
            payload=data if isinstance(data, Mapping) else None,
        )

    def health_check(self) -> Mapping[str, Any]:
        """Return the list of available models from the API."""

        url = self._models_base_url
        response = self._client.get(url, headers=self._headers)
        if response.status_code == 429:
            self._handle_error(response)
        self._handle_error(response)
        return self._extract_json(response)
