"""Google Sheets-backed catalog service."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Iterable, List
from urllib.parse import parse_qs, urlparse

import httpx

from app.models import GlassModel
from app.services.catalog_base import CatalogError, CatalogService, CatalogSnapshot
from app.utils.drive import DriveFolderUrlError, DriveUrlError, drive_view_to_direct

LOGGER = logging.getLogger("loop_bot.catalog.google")


@dataclass(slots=True)
class GoogleCatalogConfig:
    """Configuration for the Google Sheets catalog."""

    csv_url: str
    cache_ttl_seconds: int = 60
    retries: int = 3
    backoff_base: float = 0.5


class NotCSVError(CatalogError):
    """Raised when downloaded content is not a CSV."""


class GoogleSheetCatalog(CatalogService):
    """Retrieve catalog data from a published Google Sheet CSV."""

    def __init__(
        self,
        config: GoogleCatalogConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        if client is None:
            timeout = httpx.Timeout(15.0, connect=15.0, read=15.0)
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Loop/1.0",
                    "Accept": "text/csv,*/*;q=0.1",
                },
            )
        else:
            self._client = client
        self._client_owner = client is None
        self._cache: tuple[float, CatalogSnapshot] | None = None
        self._lock = asyncio.Lock()

    async def list_by_gender(self, gender: str) -> list[GlassModel]:
        snapshot = await self.snapshot()
        models = snapshot.models
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
        normalized = _normalize_gender(gender)
        gender_pool = [
            model for model in candidates if model.gender == normalized and model.unique_id not in seen
        ]
        unisex_pool = [
            model for model in candidates if model.gender == "Унисекс" and model.unique_id not in seen
        ]
        picks: list[GlassModel] = []
        max_batch = 3
        if normalized == "Унисекс":
            picks.extend(unisex_pool[:max_batch])
        else:
            picks.extend(gender_pool[:2])
            remaining = max_batch - len(picks)
            if remaining > 0:
                picks.extend(unisex_pool[:remaining])
            remaining = max_batch - len(picks)
            if remaining > 0:
                fallback_pool = [model for model in gender_pool if model not in picks]
                fallback_pool.extend(model for model in unisex_pool if model not in picks)
                picks.extend(fallback_pool[:remaining])
        LOGGER.info("Catalog returned %s models for gender=%s", len(picks), gender)
        return picks

    async def aclose(self) -> None:  # noqa: D401 - inherited docstring
        if self._client_owner:
            await self._client.aclose()

    async def snapshot(self) -> CatalogSnapshot:
        snapshot = await self._load_snapshot()
        return CatalogSnapshot(models=list(snapshot.models), version_hash=snapshot.version_hash)

    async def _load_snapshot(self) -> CatalogSnapshot:
        async with self._lock:
            if self._cache and (time.monotonic() - self._cache[0] < self._config.cache_ttl_seconds):
                cached_snapshot = self._cache[1]
                return CatalogSnapshot(
                    models=list(cached_snapshot.models),
                    version_hash=cached_snapshot.version_hash,
                )
            csv_content = await self._fetch_csv()
            models = self._parse_csv(csv_content)
            version_hash = _compute_version_hash(models)
            snapshot = CatalogSnapshot(models=list(models), version_hash=version_hash)
            self._cache = (time.monotonic(), snapshot)
            LOGGER.info(
                "Catalog cache refreshed with %s entries (payload %s bytes, hash=%s)",
                len(models),
                len(csv_content.encode("utf-8")),
                version_hash,
            )
            return CatalogSnapshot(models=list(models), version_hash=version_hash)

    async def _fetch_csv(self) -> str:
        urls: List[str] = [self._config.csv_url]
        fallback_urls = _build_fallback_urls(self._config.csv_url)
        if fallback_urls:
            urls.extend(fallback_urls)

        last_error: Exception | None = None
        for index, url in enumerate(urls):
            try:
                return await self._fetch_with_retries(url)
            except CatalogError as exc:
                last_error = exc
                if index < len(urls) - 1:
                    LOGGER.warning(
                        "Primary catalog URL failed (%s). Falling back to %s",
                        exc,
                        urls[index + 1],
                    )
                    continue
                LOGGER.error("Failed to fetch catalog CSV after fallbacks: %s", exc)
                raise
        raise CatalogError(f"Failed to fetch catalog CSV: {last_error}")

    async def _fetch_with_retries(self, url: str) -> str:
        delay = self._config.backoff_base
        last_error: Exception | None = None
        for attempt in range(1, self._config.retries + 1):
            try:
                response = await self._client.get(url)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Network error fetching CSV (attempt %s/%s): %s",
                    attempt,
                    self._config.retries,
                    exc,
                )
                if attempt < self._config.retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 2.0)
                    continue
                break

            status = response.status_code
            if 200 <= status < 300:
                self._log_redirect_chain(url, response)
                text = response.text
                if _looks_like_html(text):
                    last_error = NotCSVError("Not CSV content received")
                    LOGGER.warning(
                        "Received non-CSV content for %s (attempt %s/%s)",
                        url,
                        attempt,
                        self._config.retries,
                    )
                else:
                    LOGGER.info("Fetched catalog CSV from %s", response.url)
                    return text
            elif status == 429 or status >= 500:
                last_error = CatalogError(f"Server responded with status {status}")
                LOGGER.warning(
                    "Server error fetching CSV (status %s, attempt %s/%s)",
                    status,
                    attempt,
                    self._config.retries,
                )
            else:
                raise CatalogError(f"Unexpected status {status} fetching CSV from {url}")

            if attempt < self._config.retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

        if last_error:
            raise CatalogError(str(last_error))
        raise CatalogError("Failed to fetch catalog CSV")

    def _log_redirect_chain(self, requested_url: str, response: httpx.Response) -> None:
        if not response.history:
            LOGGER.debug("Catalog fetch URL %s responded without redirects", requested_url)
            return
        chain = [requested_url]
        for redirect_response in response.history:
            location = redirect_response.headers.get("location")
            if location:
                chain.append(location)
        chain.append(str(response.url))
        LOGGER.debug("Catalog fetch redirect chain: %s", " -> ".join(chain))

    def _parse_csv(self, csv_content: str) -> list[GlassModel]:
        stream = io.StringIO(csv_content)
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise CatalogError("Catalog CSV has no header row")
        models: list[GlassModel] = []
        total_rows = 0
        skipped_empty = 0
        skipped_folder = 0
        skipped_invalid = 0
        for row_index, row in enumerate(reader, start=2):
            total_rows += 1
            normalized_row = {_normalize_header(key): (value or "").strip() for key, value in row.items()}
            title = normalized_row.get("название")
            site_url = normalized_row.get("ссылка на сайт")
            if not title or not site_url:
                skipped_invalid += 1
                LOGGER.warning("Skipping row %s due to missing title or site URL", row_index)
                continue
            model_code = normalized_row.get("модель", "")
            gender_value = normalized_row.get("пол", "")
            gender = _normalize_gender(gender_value)
            img_user_original = _clean_drive_url(normalized_row.get("ссылка на изображение для пользователя", ""))
            if not img_user_original:
                skipped_empty += 1
                LOGGER.debug("Skipping row %s due to empty user image URL", row_index)
                continue
            try:
                img_user_url = drive_view_to_direct(img_user_original, export="view")
            except DriveFolderUrlError:
                skipped_folder += 1
                LOGGER.warning(
                    "Skipping row %s due to folder Drive URL; expected file share like /file/d/.../view",
                    row_index,
                )
                continue
            except DriveUrlError as exc:
                skipped_invalid += 1
                LOGGER.warning("Skipping row %s due to invalid Drive URL: %s", row_index, exc)
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

        LOGGER.info(
            "Catalog CSV parsed: total_rows=%s valid_rows=%s skipped_empty=%s skipped_folder=%s skipped_invalid=%s",
            total_rows,
            len(models),
            skipped_empty,
            skipped_folder,
            skipped_invalid,
        )
        return models


def _normalize_header(header: str) -> str:
    return header.strip().lower()


def _normalize_gender(value: str) -> str:
    prepared = (value or "").strip().lower()
    male_tokens = {"муж", "мужской", "м", "male", "m"}
    female_tokens = {"жен", "женский", "ж", "female", "f"}
    unisex_tokens = {"унисекс", "uni", "unisex", "u"}
    if prepared in male_tokens or prepared.startswith("муж"):
        return "Мужской"
    if prepared in female_tokens or prepared.startswith("жен"):
        return "Женский"
    if prepared in unisex_tokens or prepared.startswith("уни") or prepared.startswith("uni"):
        return "Унисекс"
    if not prepared:
        return "Унисекс"
    if prepared:
        LOGGER.warning("Unknown gender value '%s' in catalog, treating as Other", value)
    return "Other"


def _make_fallback_id(title: str, site_url: str) -> str:
    digest = hashlib.sha256(f"{title}|{site_url}".encode("utf-8")).hexdigest()
    return digest[:16]


def _looks_like_html(payload: str) -> bool:
    sample = payload.lstrip()[:256].lower()
    return "<html" in sample or "<!doctype" in sample


def _build_fallback_urls(original_url: str) -> list[str]:
    parsed = urlparse(original_url)
    if parsed.netloc != "docs.google.com":
        return []
    parts = [segment for segment in parsed.path.split("/") if segment]
    if "spreadsheets" not in parts:
        return []
    try:
        d_index = parts.index("d")
    except ValueError:
        return []
    if len(parts) <= d_index + 1:
        return []
    sheet_id = parts[d_index + 1]
    if sheet_id == "e" and len(parts) > d_index + 2:
        sheet_id = parts[d_index + 2]
    if not sheet_id:
        return []
    query = parse_qs(parsed.query)
    gid = query.get("gid", ["0"])[0]
    base = "https://docs.google.com/spreadsheets/d"
    gviz = f"{base}/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    export = f"{base}/{sheet_id}/export?format=csv&gid={gid}"
    return [gviz, export]


def _clean_drive_url(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip("\u200b\u200c\u200d\ufeff")


def _compute_version_hash(models: list[GlassModel]) -> str:
    lines = []
    for model in models:
        normalized_gender = (model.gender or "").strip().lower()
        line = "|".join(
            [
                model.unique_id.strip(),
                model.title.strip(),
                model.site_url.strip(),
                model.img_user_url.strip(),
                normalized_gender,
            ]
        )
        lines.append(line)
    payload = "\n".join(lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
