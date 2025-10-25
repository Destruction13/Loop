"""Google Sheets-backed catalog service."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Iterable

import httpx

from app.models import GlassModel
from app.services.catalog_base import CatalogError, CatalogService
from app.utils.drive import drive_view_to_direct

LOGGER = logging.getLogger("loop_bot.catalog.google")


@dataclass(slots=True)
class GoogleCatalogConfig:
    """Configuration for the Google Sheets catalog."""

    csv_url: str
    cache_ttl_seconds: int = 60
    retries: int = 3
    backoff_base: float = 0.5


class GoogleSheetCatalog(CatalogService):
    """Retrieve catalog data from a published Google Sheet CSV."""

    def __init__(
        self,
        config: GoogleCatalogConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(timeout=15.0)
        self._client_owner = client is None
        self._cache: tuple[float, list[GlassModel]] | None = None
        self._lock = asyncio.Lock()

    async def list_by_gender(self, gender: str) -> list[GlassModel]:
        models = await self._load_models()
        normalized = _normalize_gender(gender)
        allowed: set[str]
        if normalized == "Унисекс":
            allowed = {"Унисекс"}
        else:
            allowed = {normalized, "Унисекс"}
        return [model for model in models if model.gender in allowed]

    async def pick_four(self, gender: str, seen_ids: Iterable[str]) -> list[GlassModel]:
        candidates = await self.list_by_gender(gender)
        seen = set(seen_ids)
        unseen = [model for model in candidates if model.unique_id not in seen]
        picked: list[GlassModel] = unseen[:4]
        if len(picked) < 4:
            fallback = [model for model in candidates if model.unique_id not in {m.unique_id for m in picked}]
            for model in fallback:
                if len(picked) >= 4:
                    break
                picked.append(model)
        LOGGER.info("Catalog returned %s models for gender=%s", len(picked), gender)
        return picked

    async def aclose(self) -> None:  # noqa: D401 - inherited docstring
        if self._client_owner:
            await self._client.aclose()

    async def _load_models(self) -> list[GlassModel]:
        async with self._lock:
            if self._cache and (time.monotonic() - self._cache[0] < self._config.cache_ttl_seconds):
                return list(self._cache[1])
            csv_content = await self._fetch_csv()
            models = self._parse_csv(csv_content)
            self._cache = (time.monotonic(), models)
            LOGGER.info("Catalog cache refreshed with %s entries", len(models))
            return list(models)

    async def _fetch_csv(self) -> str:
        delay = self._config.backoff_base
        last_error: Exception | None = None
        for attempt in range(1, self._config.retries + 1):
            try:
                response = await self._client.get(self._config.csv_url)
                response.raise_for_status()
                LOGGER.info("Fetched catalog CSV on attempt %s", attempt)
                return response.text
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:  # pragma: no cover - network errors
                last_error = exc
                LOGGER.warning("Failed to fetch CSV (attempt %s/%s): %s", attempt, self._config.retries, exc)
                if attempt < self._config.retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 2.0)
        raise CatalogError(f"Failed to fetch catalog CSV: {last_error}")

    def _parse_csv(self, csv_content: str) -> list[GlassModel]:
        stream = io.StringIO(csv_content)
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise CatalogError("Catalog CSV has no header row")
        models: list[GlassModel] = []
        for row_index, row in enumerate(reader, start=2):
            normalized_row = {_normalize_header(key): (value or "").strip() for key, value in row.items()}
            title = normalized_row.get("название")
            site_url = normalized_row.get("ссылка на сайт")
            if not title or not site_url:
                LOGGER.warning("Skipping row %s due to missing title or site URL", row_index)
                continue
            model_code = normalized_row.get("модель", "")
            gender_value = normalized_row.get("пол", "")
            gender = _normalize_gender(gender_value)
            img_user_original = normalized_row.get("ссылка на изображение для пользователя", "")
            if not img_user_original:
                LOGGER.warning("Skipping row %s due to missing user image URL", row_index)
                continue
            try:
                img_user_url = drive_view_to_direct(img_user_original, export="view")
            except ValueError as exc:
                LOGGER.warning("Skipping row %s due to invalid drive URL: %s", row_index, exc)
                continue
            img_nano_url = normalized_row.get("ссылка на изображение для nanobanana", "")
            unique_id = normalized_row.get("уникальный id") or _make_fallback_id(title, site_url)
            model = GlassModel(
                unique_id=unique_id,
                title=title,
                model_code=model_code,
                site_url=site_url,
                img_user_url=img_user_url,
                img_nano_url=img_nano_url,
                gender=gender,
            )
            models.append(model)
        return models


def _normalize_header(header: str) -> str:
    return header.strip().lower()


def _normalize_gender(value: str) -> str:
    prepared = (value or "").strip().lower()
    if prepared.startswith("муж") or prepared.startswith("male"):
        return "Мужской"
    if prepared.startswith("жен") or prepared.startswith("female"):
        return "Женский"
    if prepared.startswith("уни") or prepared.startswith("uni"):
        return "Унисекс"
    return "Унисекс"


def _make_fallback_id(title: str, site_url: str) -> str:
    digest = hashlib.sha256(f"{title}|{site_url}".encode("utf-8")).hexdigest()
    return digest[:16]
