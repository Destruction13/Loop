"""Helpers for loading event scenes from Google Sheets CSV."""

from __future__ import annotations

import asyncio
import csv
import io
import time
from dataclasses import dataclass
from typing import Final, Iterable
from urllib.parse import parse_qs, urlparse

import httpx

from logger import get_logger

LOGGER = get_logger("event.scenes")

_CSV_INDICATORS: Final = ("output=csv", "tqx=out:csv", "format=csv")


@dataclass(slots=True)
class EventScene:
    scene_id: int
    drive_url: str
    gender: str | None = None


class EventScenesService:
    """Fetch and cache event scenes from a Google Sheets source."""

    def __init__(
        self,
        sheet_url: str,
        *,
        cache_ttl_seconds: int = 60,
        retries: int = 3,
        backoff_base: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._sheet_url = (sheet_url or "").strip()
        self._cache_ttl = max(cache_ttl_seconds, 1)
        self._retries = max(retries, 1)
        self._backoff_base = max(backoff_base, 0.1)
        if client is None:
            timeout = httpx.Timeout(15.0, connect=15.0, read=15.0)
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Loop/1.0", "Accept": "text/csv,*/*;q=0.1"},
            )
            self._client_owner = True
        else:
            self._client = client
            self._client_owner = False
        self._cache: tuple[float, list[EventScene]] | None = None
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._client_owner:
            await self._client.aclose()

    async def list_scenes(self) -> list[EventScene]:
        cached = self._cache
        if cached and time.time() - cached[0] <= self._cache_ttl:
            return list(cached[1])
        async with self._lock:
            cached = self._cache
            if cached and time.time() - cached[0] <= self._cache_ttl:
                return list(cached[1])
            csv_text = await self._fetch_csv()
            scenes = _parse_scenes(csv_text)
            self._cache = (time.time(), scenes)
            return list(scenes)

    async def _fetch_csv(self) -> str:
        candidates = _build_csv_urls(self._sheet_url)
        if not candidates:
            raise RuntimeError("Event scenes sheet URL is not configured")
        last_exc: Exception | None = None
        for url in candidates:
            for attempt in range(1, self._retries + 1):
                try:
                    response = await self._client.get(url)
                    response.raise_for_status()
                    return response.text
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < self._retries:
                        await asyncio.sleep(self._backoff_base * attempt)
                        continue
                    LOGGER.debug(
                        "Failed to fetch event scenes CSV from %s: %s",
                        url,
                        exc,
                    )
        raise RuntimeError("Failed to load event scenes CSV") from last_exc


def _parse_scenes(csv_text: str) -> list[EventScene]:
    stream = io.StringIO(csv_text)
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        raise RuntimeError("Event scenes CSV has no headers")
    headers = {name.strip().casefold(): name for name in reader.fieldnames}
    id_key = headers.get("id")
    scene_key = headers.get("\u0441\u044e\u0436\u0435\u0442\u044b") or headers.get("scene")
    gender_key = headers.get("\u043f\u043e\u043b") or headers.get("gender")
    if not id_key or not scene_key:
        raise RuntimeError("Event scenes CSV must contain id and \u0441\u044e\u0436\u0435\u0442\u044b columns")
    scenes: list[EventScene] = []
    for row in reader:
        raw_id = (row.get(id_key) or "").strip()
        raw_url = (row.get(scene_key) or "").strip()
        if not raw_id or not raw_url:
            continue
        raw_gender = (row.get(gender_key) or "").strip() if gender_key else ""
        try:
            scene_id = int(raw_id)
        except ValueError:
            continue
        gender = raw_gender or None
        scenes.append(EventScene(scene_id=scene_id, drive_url=raw_url, gender=gender))
    if not scenes:
        raise RuntimeError("Event scenes CSV is empty")
    return scenes


def _build_csv_urls(sheet_url: str) -> list[str]:
    sanitized = (sheet_url or "").strip()
    if not sanitized:
        return []
    lowered = sanitized.lower()
    if any(token in lowered for token in _CSV_INDICATORS):
        return [sanitized]
    sheet_id, gid = _extract_sheet_id_and_gid(sanitized)
    if sheet_id:
        return _build_urls_from_sheet_id(sheet_id, gid)
    return [sanitized]


def _extract_sheet_id_and_gid(url_or_id: str) -> tuple[str | None, str | None]:
    if not url_or_id:
        return None, None
    s = url_or_id.strip().strip('"').strip("'")
    if "/" not in s and "?" not in s:
        return s, None
    try:
        parsed = urlparse(s)
        match = None
        if "/spreadsheets/d/" in parsed.path:
            parts = parsed.path.split("/")
            if "d" in parts:
                idx = parts.index("d")
                if idx + 1 < len(parts):
                    match = parts[idx + 1]
        query = parse_qs(parsed.query)
        gid = (query.get("gid") or [None])[0]
        return match, gid
    except Exception:
        return None, None


def _build_urls_from_sheet_id(sheet_id: str, gid: str | None) -> list[str]:
    sanitized_gid = (gid or "0").strip() or "0"
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    gviz = f"{base}/gviz/tq?tqx=out:csv&gid={sanitized_gid}"
    export = f"{base}/export?format=csv&gid={sanitized_gid}"
    return [gviz, export]


__all__ = ["EventScene", "EventScenesService"]
