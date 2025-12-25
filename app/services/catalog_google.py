"""Google Sheets-backed catalog service."""

from __future__ import annotations

import asyncio
import csv
from collections import Counter
import hashlib
import io
import random
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qs, urlparse, urlsplit

import httpx

from app.models import GlassModel, STYLE_UNKNOWN
from app.services.catalog_base import (
    CatalogBatch,
    CatalogError,
    CatalogService,
    CatalogSnapshot,
)
from app.utils.drive import DriveFolderUrlError, DriveUrlError, drive_view_to_direct
from logger import get_logger, info_domain

LOGGER = get_logger("sheets.catalog")

_SOURCE_LABELS = {
    "direct-url": "прямой CSV",
    "doc-id": "ID таблицы",
    "configured-url": "настроенный URL",
}


def _normalize_header(value: str) -> str:
    return value.strip().lower()


HEADER_SYNONYMS: dict[str, frozenset[str]] = {
    "unique_id": frozenset(
        _normalize_header(token)
        for token in ("уникальный id", "id", "uid", "unique id", "unique_id")
    ),
    "title": frozenset(
        _normalize_header(token)
        for token in ("название", "наименование", "title", "name")
    ),
    "model": frozenset(
        _normalize_header(token) for token in ("модель", "model")
    ),
    "site": frozenset(
        _normalize_header(token)
        for token in ("ссылка на сайт", "url", "site", "ссылка")
    ),
    "img_nb": frozenset(
        _normalize_header(token)
        for token in (
            "ссылка на изображение для nanobanana",
            "nanobanana",
            "nb image",
            "image_nb",
            "img_nb",
        )
    ),
    "img_user": frozenset(
        _normalize_header(token)
        for token in (
            "ссылка на изображение для пользователя",
            "user image",
            "image_user",
            "img_user",
        )
    ),
    "gender": frozenset(
        _normalize_header(token) for token in ("пол", "gender", "sex")
    ),
}

REQUIRED_HEADERS = frozenset(
    ("unique_id", "title", "model", "site", "img_nb", "img_user", "gender")
)

STYLE_HEADER_TOKEN = _normalize_header("Стили")


@dataclass(slots=True)
class GoogleCatalogConfig:
    """Configuration for the Google Sheets catalog."""

    csv_url: str | None = None
    sheet_id: str | None = None
    sheet_gid: str | None = None
    cache_ttl_seconds: int = 60
    retries: int = 3
    backoff_base: float = 0.5
    parse_row_limit: int | None = None


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
        self._source_logged = False

    async def list_by_gender(self, gender: str) -> list[GlassModel]:
        snapshot = await self.snapshot()
        models = snapshot.models
        normalized = _normalize_gender(gender)
        allowed: set[str]
        if normalized == "unisex":
            allowed = {"unisex"}
        elif normalized in {"male", "female"}:
            allowed = {normalized, "unisex"}
        else:
            allowed = {normalized, "unisex"}
        return [model for model in models if model.gender in allowed]

    async def pick_batch(
        self,
        *,
        gender: str,
        batch_size: int,
        scheme: str,
        rng: random.Random | None = None,
        snapshot: CatalogSnapshot | None = None,
    ) -> CatalogBatch:
        rng = rng or random.Random()
        snapshot = snapshot or await self.snapshot()

        unique_models: dict[str, GlassModel] = {}
        for model in snapshot.models:
            unique_models.setdefault(model.unique_id, model)
        models = list(unique_models.values())

        normalized_gender = _normalize_gender(gender)
        normalized_scheme = (scheme or "GENDER_AND_UNISEX_ONLY").strip().upper()
        gender_pool: list[GlassModel] = []
        unisex_pool: list[GlassModel] = []

        for model in models:
            group = _normalize_gender(model.gender)
            if group == normalized_gender:
                gender_pool.append(model)
            elif group == "unisex":
                unisex_pool.append(model)

        if normalized_scheme == "UNIVERSAL":
            picks, exhausted = _pick_universal_batch(
                rng,
                gender_pool,
                unisex_pool,
                batch_size,
                normalized_gender,
            )
        elif normalized_gender == "unisex":
            picks, exhausted = _pick_unisex_batch(rng, unisex_pool, batch_size)
        else:
            picks, exhausted = _pick_gender_batch(
                rng, gender_pool, unisex_pool, batch_size, normalized_scheme
            )

        LOGGER.debug(
            "Catalog returned batch items=%s exhausted=%s for gender=%s",
            [model.unique_id for model in picks],
            exhausted,
            gender,
        )
        return CatalogBatch(items=picks, exhausted=exhausted)

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
            try:
                csv_content = await self._fetch_csv()
                models = self._parse_csv(csv_content)
            except CatalogError as exc:
                LOGGER.error(
                    "Не удалось загрузить или разобрать каталог: %s",
                    exc,
                    extra={"stage": "SHEET_PARSE_ERROR"},
                )
                raise

            version_hash = _compute_version_hash(models)
            snapshot = CatalogSnapshot(models=list(models), version_hash=version_hash)
            self._cache = (time.monotonic(), snapshot)
            LOGGER.debug(
                "Каталог обновлён: %s записей (хэш %s)",
                len(models),
                version_hash,
            )
            info_domain(
                "sheets.load",
                f"Таблица подгружена — {len(models)} строк",
                stage="SHEET_LOADED",
                rows=len(models),
                hash=version_hash,
            )
            info_domain(
                "sheets.load",
                f"Парсинг каталога ok — {len(models)} строк",
                stage="SHEET_PARSE_OK",
                rows=len(models),
            )
            return CatalogSnapshot(models=list(models), version_hash=version_hash)

    async def _fetch_csv(self) -> str:
        urls, source = _resolve_fetch_plan(self._config)
        LOGGER.debug("Catalog source=%s", source)
        if not self._source_logged:
            source_label = _SOURCE_LABELS.get(source, source)
            info_domain(
                "sheets.load",
                f"Источник каталога: {source_label}",
                stage="SHEET_SOURCE",
                source=source,
            )
            self._source_logged = True
        LOGGER.debug("Catalog url candidates=%s", ", ".join(urls))

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
                LOGGER.debug(
                    "Catalog fetch attempt %s/%s url=%s",
                    attempt,
                    self._config.retries,
                    url,
                )
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
            LOGGER.debug("Catalog fetch status=%s url=%s", status, response.url)
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
                    LOGGER.debug("Fetched catalog CSV from %s", response.url)
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

        header_map = _resolve_header_map(reader.fieldnames)
        style_header = header_map.get("style")
        missing_headers = REQUIRED_HEADERS - set(header_map)
        if missing_headers:
            missing = ", ".join(sorted(missing_headers))
            raise CatalogError(f"Catalog CSV missing required columns: {missing}")
        if style_header is None:
            LOGGER.warning(
                "Catalog CSV missing style column 'Стили'; defaulting styles to %s",
                STYLE_UNKNOWN,
            )

        models: list[GlassModel] = []
        total_rows = 0
        skipped_empty = 0
        skipped_folder = 0
        skipped_invalid = 0
        limit = self._config.parse_row_limit or 0

        for row_index, row in enumerate(reader, start=2):
            total_rows += 1
            if not any((value or "").strip() for value in row.values()):
                skipped_empty += 1
                LOGGER.debug("Skipping row %s because it is empty", row_index)
                continue

            normalized_row = {
                key: (row.get(source) or "").strip()
                for key, source in header_map.items()
            }

            unique_id = normalized_row["unique_id"]
            title = normalized_row["title"]
            model_code = normalized_row["model"]
            site_url = normalized_row["site"]
            img_nb_value = _clean_drive_url(normalized_row["img_nb"])
            img_user_value = _clean_drive_url(normalized_row["img_user"])
            gender_raw = normalized_row["gender"]
            style_raw = normalized_row.get("style", "")
            style = style_raw or STYLE_UNKNOWN

            required_values = {
                "unique_id": unique_id,
                "title": title,
                "model": model_code,
                "site": site_url,
                "img_nb": img_nb_value,
                "img_user": img_user_value,
                "gender": gender_raw,
            }
            if any(not value for value in required_values.values()):
                missing = [name for name, value in required_values.items() if not value]
                skipped_invalid += 1
                LOGGER.warning(
                    "Skipping row %s due to empty values in columns: %s",
                    row_index,
                    ", ".join(missing),
                )
                continue

            if not _is_valid_http_url(site_url):
                skipped_invalid += 1
                LOGGER.warning("Skipping row %s due to invalid site URL: %s", row_index, site_url)
                continue

            if not _is_valid_http_url(img_nb_value):
                skipped_invalid += 1
                LOGGER.warning(
                    "Skipping row %s due to invalid NanoBanana image URL: %s",
                    row_index,
                    img_nb_value,
                )
                continue

            if _is_drive_folder_link(img_user_value) or _is_drive_folder_link(img_nb_value):
                skipped_folder += 1
                LOGGER.warning(
                    "Skipping row %s because one of the image URLs points to a Drive folder",
                    row_index,
                )
                continue

            try:
                img_user_url = drive_view_to_direct(img_user_value, export="view")
            except DriveFolderUrlError:
                skipped_folder += 1
                LOGGER.warning(
                    "Skipping row %s due to Drive folder in user image URL", row_index
                )
                continue
            except DriveUrlError as exc:
                skipped_invalid += 1
                LOGGER.warning(
                    "Skipping row %s due to invalid user image Drive URL: %s",
                    row_index,
                    exc,
                )
                continue

            gender = _normalize_gender(gender_raw)
            if gender not in {"male", "female", "unisex"}:
                skipped_invalid += 1
                LOGGER.warning(
                    "Skipping row %s due to unsupported gender value: %s",
                    row_index,
                    gender_raw,
                )
                continue

            model = GlassModel(
                unique_id=unique_id,
                title=title,
                model_code=model_code,
                site_url=site_url,
                img_user_url=img_user_url,
                img_nano_url=img_nb_value,
                gender=gender,
                style=style,
            )
            models.append(model)

            if limit and len(models) >= limit:
                LOGGER.info(
                    "Catalog CSV row limit %s reached at data row %s, stopping parse",
                    limit,
                    row_index,
                )
                break

        LOGGER.info(
            "Catalog CSV parsed: total_rows=%s valid_rows=%s skipped_empty=%s skipped_folder=%s skipped_invalid=%s",
            total_rows,
            len(models),
            skipped_empty,
            skipped_folder,
            skipped_invalid,
        )
        if models:
            style_counts = Counter(model.style for model in models)
            known_count = sum(
                count for style, count in style_counts.items() if style != STYLE_UNKNOWN
            )
            top_styles = [
                f"{style}:{count}"
                for style, count in style_counts.most_common(5)
                if style != STYLE_UNKNOWN
            ]
            LOGGER.info(
                "Catalog styles: total=%s known=%s top=%s",
                len(models),
                known_count,
                ", ".join(top_styles) if top_styles else "none",
            )
        return models


def _resolve_header_map(fieldnames: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    style_header = None
    for name in fieldnames:
        normalized = _normalize_header(name)
        if normalized == STYLE_HEADER_TOKEN and style_header is None:
            style_header = name
        for canonical, synonyms in HEADER_SYNONYMS.items():
            if normalized in synonyms and canonical not in mapping:
                mapping[canonical] = name
                break
    if style_header is not None:
        mapping["style"] = style_header
    return mapping


def _is_valid_http_url(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_drive_folder_link(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return "drive.google.com/drive/folders" in normalized


def _normalize_gender(value: str) -> str:
    prepared = (value or "").strip().lower()
    if not prepared:
        return "other"

    male_tokens = {"муж", "мужской", "мужчина", "м", "male", "m"}
    female_tokens = {"жен", "женский", "женщина", "ж", "female", "f"}
    unisex_tokens = {"унисекс", "uni", "unisex", "u"}

    if prepared in male_tokens or prepared.startswith("муж"):
        return "male"
    if prepared in female_tokens or prepared.startswith("жен"):
        return "female"
    if prepared in unisex_tokens or prepared.startswith("уни") or prepared.startswith("uni"):
        return "unisex"
    if prepared in {"other", "другое"}:
        return "other"

    LOGGER.warning("Unknown gender value '%s' in catalog, treating as other", value)
    return "other"


def _sample(
    rng: random.Random, items: Iterable[GlassModel], count: int
) -> list[GlassModel]:
    pool = list(items)
    if count <= 0:
        return []
    if len(pool) <= count:
        return list(pool)
    return rng.sample(pool, count)


def _pick_unisex_batch(
    rng: random.Random, pool: list[GlassModel], batch_size: int
) -> tuple[list[GlassModel], bool]:
    selection = _sample(rng, pool, min(batch_size, len(pool)))
    exhausted = len(selection) == 0
    return selection, exhausted


def _pick_gender_batch(
    rng: random.Random,
    gender_pool: list[GlassModel],
    unisex_pool: list[GlassModel],
    batch_size: int,
    scheme: str,
) -> tuple[list[GlassModel], bool]:
    normalized_scheme = (scheme or "GENDER_AND_UNISEX_ONLY").strip().upper()
    picks: list[GlassModel] = []
    used_ids: set[str] = set()

    schemes: list[str] = []
    if len(gender_pool) >= 2:
        schemes.append("GG")
    if len(gender_pool) >= 1 and len(unisex_pool) >= 1:
        schemes.append("GU")

    chosen_scheme: str | None = None
    if normalized_scheme == "GENDER_AND_GENDER_ONLY":
        if "GG" in schemes:
            chosen_scheme = "GG"
    elif normalized_scheme == "GENDER_AND_UNISEX_ONLY":
        if "GU" in schemes:
            chosen_scheme = "GU"
    elif normalized_scheme == "GENDER_OR_GENDER_UNISEX" and schemes:
        if len(schemes) == 1:
            chosen_scheme = schemes[0]
        else:
            chosen_scheme = rng.choice(schemes)

    if chosen_scheme == "GG":
        selection = _sample(rng, gender_pool, min(2, len(gender_pool)))
        picks.extend(selection)
        used_ids.update(model.unique_id for model in selection)
    elif chosen_scheme == "GU":
        gender_selection = _sample(rng, gender_pool, 1)
        picks.extend(gender_selection)
        used_ids.update(model.unique_id for model in gender_selection)
        remaining_unisex = [
            model for model in unisex_pool if model.unique_id not in used_ids
        ]
        unisex_selection = _sample(rng, remaining_unisex, 1)
        picks.extend(unisex_selection)
        used_ids.update(model.unique_id for model in unisex_selection)
    else:
        if gender_pool:
            gender_selection = _sample(
                rng, gender_pool, min(batch_size, len(gender_pool))
            )
            picks.extend(gender_selection)
            used_ids.update(model.unique_id for model in gender_selection)
        elif unisex_pool:
            unisex_selection = _sample(
                rng, unisex_pool, min(batch_size, len(unisex_pool))
            )
            picks.extend(unisex_selection)
            used_ids.update(model.unique_id for model in unisex_selection)

    if len(picks) < batch_size and unisex_pool:
        remaining_slots = batch_size - len(picks)
        remaining_unisex = [
            model for model in unisex_pool if model.unique_id not in used_ids
        ]
        if remaining_unisex:
            unisex_selection = _sample(
                rng, remaining_unisex, min(remaining_slots, len(remaining_unisex))
            )
            picks.extend(unisex_selection)
            used_ids.update(model.unique_id for model in unisex_selection)

    if len(picks) < batch_size and gender_pool:
        remaining_slots = batch_size - len(picks)
        remaining_gender = [
            model for model in gender_pool if model.unique_id not in used_ids
        ]
        if remaining_gender:
            gender_selection = _sample(
                rng, remaining_gender, min(remaining_slots, len(remaining_gender))
            )
            picks.extend(gender_selection)
            used_ids.update(model.unique_id for model in gender_selection)

    exhausted = len(picks) == 0
    return picks, exhausted


def _pick_universal_batch(
    rng: random.Random,
    gender_pool: list[GlassModel],
    unisex_pool: list[GlassModel],
    batch_size: int,
    normalized_gender: str,
) -> tuple[list[GlassModel], bool]:
    if normalized_gender == "unisex":
        allowed = list(unisex_pool)
    else:
        allowed = list(gender_pool)
        allowed.extend(unisex_pool)

    selection = _sample(rng, allowed, min(batch_size, len(allowed)))
    exhausted = len(selection) == 0
    return selection, exhausted


def _looks_like_html(payload: str) -> bool:
    sample = payload.lstrip()[:256].lower()
    return "<html" in sample or "<!doctype" in sample


def _build_fallback_urls(original_url: str, *, default_gid: str | None = None) -> list[str]:
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
    gid = query.get("gid", [default_gid or "0"])[0]
    return _build_urls_from_sheet_id(sheet_id, gid)


def _build_urls_from_sheet_id(sheet_id: str, gid: str) -> list[str]:
    base = "https://docs.google.com/spreadsheets/d"
    gviz = f"{base}/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    export = f"{base}/{sheet_id}/export?format=csv&gid={gid}"
    return [gviz, export]


def _resolve_fetch_plan(config: GoogleCatalogConfig) -> tuple[list[str], str]:
    csv_url = (config.csv_url or "").strip()
    sheet_id = (config.sheet_id or "").strip()
    gid = (config.sheet_gid or "").strip() or "0"

    if csv_url:
        if _is_direct_csv_url(csv_url):
            return [csv_url], "direct-url"
        fallback_urls = _build_fallback_urls(csv_url, default_gid=gid)
        if fallback_urls:
            return fallback_urls, "doc-id"
        return [csv_url], "configured-url"

    if sheet_id:
        return _build_urls_from_sheet_id(sheet_id, gid), "doc-id"

    raise CatalogError("Catalog CSV source is not configured")


def _is_direct_csv_url(url: str) -> bool:
    lowered = url.lower()
    return (
        "output=csv" in lowered
        or "tqx=out:csv" in lowered
        or "format=csv" in lowered
    )


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
