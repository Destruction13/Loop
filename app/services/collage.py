"""Utilities for building collage previews of catalog models."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx
from PIL import Image, ImageDraw

LOGGER = logging.getLogger("loop_bot.collage")


@dataclass(frozen=True)
class _DownloadedImage:
    """Internal representation of a downloaded collage image."""

    position: int
    url: str
    data: bytes


@dataclass(slots=True)
class _PastedImage:
    """Normalized PIL image ready to be pasted."""

    image: Image.Image


@dataclass(slots=True)
class _Placement:
    """Location of an image inside the collage canvas."""

    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class CollageResult:
    """Result of a collage generation request."""

    image_bytes: bytes
    included_positions: tuple[int, ...]


class CollageService:
    """Service responsible for creating cached collages for model pairs."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_width: int,
        padding_px: int,
        cache_ttl_sec: int,
        draw_divider: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._enabled = enabled
        self._max_width = max_width
        self._padding_px = padding_px
        self._cache_ttl = cache_ttl_sec
        self._draw_divider = draw_divider
        if client is None:
            timeout = httpx.Timeout(10.0, connect=10.0, read=10.0)
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
            self._client_owner = True
        else:
            self._client = client
            self._client_owner = False
        self._cache: dict[tuple[object, ...], tuple[float, CollageResult]] = {}
        self._cache_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Return whether collage generation is enabled."""

        return self._enabled

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned."""

        if self._client_owner:
            await self._client.aclose()

    async def build_collage(
        self,
        left_image_url: str | None,
        right_image_url: str | None,
    ) -> CollageResult | None:
        """Build a collage for the provided image URLs.

        Returns ``None`` when collage generation is disabled or all downloads
        fail. The resulting ``included_positions`` reflect which inputs were
        successfully rendered (0 for left, 1 for right).
        """

        if not self._enabled:
            return None

        sources = tuple(
            (position, url)
            for position, url in enumerate((left_image_url, right_image_url))
            if url
        )
        if not sources:
            return None

        cache_key = (
            self._draw_divider,
            tuple(url for _, url in sources),
        )
        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        downloaded = await self._download_images(sources)
        if not downloaded:
            return None

        positions = tuple(image.position for image in downloaded)
        try:
            image_bytes = await asyncio.to_thread(
                self._compose_collage,
                downloaded,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to compose collage for %s: %s",
                tuple(url for _, url in sources),
                exc,
            )
            return None

        result = CollageResult(image_bytes=image_bytes, included_positions=positions)
        if len(positions) == len(sources):
            await self._store_in_cache(cache_key, result)
        return result

    async def _get_from_cache(self, key: tuple[object, ...]) -> CollageResult | None:
        now = time.monotonic()
        async with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, result = entry
            if expires_at < now:
                del self._cache[key]
                return None
            return result

    async def _store_in_cache(
        self, key: tuple[object, ...], result: CollageResult
    ) -> None:
        async with self._cache_lock:
            self._cache[key] = (time.monotonic() + self._cache_ttl, result)

    async def _download_images(
        self, sources: Sequence[tuple[int, str]]
    ) -> list[_DownloadedImage]:
        tasks = [self._client.get(url) for _, url in sources]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        downloaded: list[_DownloadedImage] = []
        for index, result in enumerate(responses):
            if isinstance(result, Exception):
                LOGGER.warning("Failed to download %s: %s", sources[index][1], result)
                continue
            if result.status_code >= 400:
                LOGGER.warning(
                    "Image request failed for %s with status %s",
                    sources[index][1],
                    result.status_code,
                )
                continue
            content_type = result.headers.get("content-type", "")
            if "image" not in content_type:
                LOGGER.warning(
                    "Skipping %s due to invalid content-type %s",
                    sources[index][1],
                    content_type,
                )
                continue
            downloaded.append(
                _DownloadedImage(
                    position=sources[index][0],
                    url=sources[index][1],
                    data=result.content,
                )
            )
        return downloaded

    def _compose_collage(self, images: Iterable[_DownloadedImage]) -> bytes:
        sources: list[_PastedImage] = []
        for downloaded in images:
            image = Image.open(io.BytesIO(downloaded.data))
            converted = image.convert("RGBA")
            sources.append(_PastedImage(image=converted))
            image.close()
        if not sources:
            raise ValueError("No images to compose")

        resized = self._normalize_heights(sources)
        collage, padding, placements = self._place_images(resized)
        if self._draw_divider and len(placements) > 1:
            self._add_dividers(collage, placements, padding)
        buffer = io.BytesIO()
        try:
            output = collage.convert("RGB")
            output.save(buffer, format="JPEG", quality=85)
        finally:
            collage.close()
            seen: set[int] = set()
            for item in resized:
                identifier = id(item.image)
                if identifier in seen:
                    continue
                item.image.close()
                seen.add(identifier)
        return buffer.getvalue()

    def _normalize_heights(self, images: list[_PastedImage]) -> list[_PastedImage]:
        target_height = max(image.image.height for image in images)
        normalized: list[_PastedImage] = []
        for item in images:
            if item.image.height == target_height:
                normalized.append(item)
                continue
            ratio = target_height / item.image.height
            new_width = max(1, int(item.image.width * ratio))
            resized = item.image.resize((new_width, target_height), Image.LANCZOS)
            normalized.append(_PastedImage(image=resized))
        return normalized

    def _place_images(
        self, images: list[_PastedImage]
    ) -> tuple[Image.Image, int, list[_Placement]]:
        padding = self._padding_px
        total_width = sum(image.image.width for image in images) + padding * (len(images) + 1)
        height = images[0].image.height if images else 0

        if total_width > self._max_width:
            scale = self._max_width / total_width
            padding = max(1, int(padding * scale))
            scaled_images: list[_PastedImage] = []
            for item in images:
                new_width = max(1, int(item.image.width * scale))
                new_height = max(1, int(item.image.height * scale))
                scaled = item.image.resize((new_width, new_height), Image.LANCZOS)
                scaled_images.append(_PastedImage(image=scaled))
            images = scaled_images
            height = images[0].image.height
            total_width = sum(image.image.width for image in images) + padding * (len(images) + 1)
        else:
            height = images[0].image.height

        canvas_height = height + padding * 2
        collage = Image.new("RGBA", (total_width, canvas_height), color=(255, 255, 255, 255))
        current_x = padding
        placements: list[_Placement] = []
        for item in images:
            y = padding + (height - item.image.height) // 2
            collage.paste(item.image, (current_x, y))
            placements.append(
                _Placement(
                    left=current_x,
                    top=y,
                    right=current_x + item.image.width,
                    bottom=y + item.image.height,
                )
            )
            current_x += item.image.width + padding
        return collage, padding, placements

    def _add_dividers(
        self,
        collage: Image.Image,
        placements: list[_Placement],
        padding: int,
    ) -> None:
        divider_color = (229, 229, 229)
        divider_width = 2
        draw = ImageDraw.Draw(collage)
        top = padding
        bottom = collage.height - padding
        for left_box, right_box in zip(placements, placements[1:]):
            gap_start = left_box.right
            gap_end = right_box.left
            if gap_end <= gap_start:
                continue
            x = gap_start + (gap_end - gap_start) // 2
            draw.line([(x, top), (x, bottom)], fill=divider_color, width=divider_width)

