"""Utilities for building collage previews of catalog models."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import httpx
import PIL
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger("loop_bot.collage")


@dataclass(frozen=True)
class CollageItem:
    """Input item describing the collage image source."""

    url: str
    display_index: int


@dataclass(frozen=True)
class _DownloadedImage:
    """Internal representation of a downloaded collage image."""

    position: int
    item: CollageItem
    data: bytes


@dataclass(slots=True)
class _PastedImage:
    """Normalized PIL image with its display index."""

    image: Image.Image
    display_index: int


@dataclass(slots=True)
class _Placement:
    """Location of an image inside the collage canvas."""

    left: int
    top: int
    right: int
    bottom: int
    display_index: int


@dataclass(frozen=True)
class CollageResult:
    """Result of a collage generation request."""

    image_bytes: bytes
    included_positions: tuple[int, ...]


class CollageService:
    """Service responsible for creating cached collages for model pairs."""

    _font_warning_emitted = False

    def __init__(
        self,
        *,
        enabled: bool,
        max_width: int,
        padding_px: int,
        cache_ttl_sec: int,
        draw_divider: bool = True,
        draw_badges: bool = True,
        index_size_px: int = 64,
        index_pad_px: int = 16,
        index_bg: str = "#000000",
        index_bg_alpha: float = 0.65,
        index_text_color: str = "#FFFFFF",
        index_text_size: int = 36,
        index_stroke: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._enabled = enabled
        self._max_width = max_width
        self._padding_px = padding_px
        self._cache_ttl = cache_ttl_sec
        self._draw_divider = draw_divider
        self._draw_badges = draw_badges
        self._index_size_px = max(1, index_size_px)
        self._index_pad_px = max(0, index_pad_px)
        self._index_text_size = max(1, index_text_size)
        self._index_stroke = max(0, index_stroke)
        self._index_bg_rgba = self._hex_to_rgba(index_bg, index_bg_alpha)
        self._index_text_rgba = self._hex_to_rgba(index_text_color, 1.0)
        self._index_stroke_rgba = (255, 255, 255, 255)
        if client is None:
            timeout = httpx.Timeout(10.0, connect=10.0, read=10.0)
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
            self._client_owner = True
        else:
            self._client = client
            self._client_owner = False
        self._cache: dict[tuple[object, ...], tuple[float, CollageResult]] = {}
        self._cache_lock = asyncio.Lock()
        self._badge_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
        self._prepare_badge_font()

    @property
    def enabled(self) -> bool:
        """Return whether collage generation is enabled."""

        return self._enabled

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned."""

        if self._client_owner:
            await self._client.aclose()

    async def build_collage(self, items: Sequence[CollageItem]) -> CollageResult | None:
        """Build a collage for the given image URLs.

        Returns ``None`` when collage generation is disabled or failed for all
        images. The order of ``urls`` is preserved in the resulting indices.
        """

        if not self._enabled:
            return None

        if not items:
            return None

        normalized = tuple((item.url, item.display_index) for item in items)
        cache_key = (
            self._draw_divider,
            self._draw_badges,
            self._index_size_px,
            self._index_pad_px,
            self._index_text_size,
            self._index_stroke,
            self._index_bg_rgba,
            self._index_text_rgba,
            normalized,
        )
        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        downloaded = await self._download_images(items)
        if not downloaded:
            return None

        positions = tuple(image.position for image in downloaded)
        try:
            image_bytes = await asyncio.to_thread(
                self._compose_collage,
                downloaded,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to compose collage for %s: %s", normalized, exc)
            return None

        result = CollageResult(image_bytes=image_bytes, included_positions=positions)
        if len(positions) == len(normalized):
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
        self, items: Sequence[CollageItem]
    ) -> list[_DownloadedImage]:
        tasks = [self._client.get(item.url) for item in items]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        downloaded: list[_DownloadedImage] = []
        for index, result in enumerate(responses):
            if isinstance(result, Exception):
                LOGGER.warning("Failed to download %s: %s", items[index].url, result)
                continue
            if result.status_code >= 400:
                LOGGER.warning(
                    "Image request failed for %s with status %s",
                    items[index].url,
                    result.status_code,
                )
                continue
            content_type = result.headers.get("content-type", "")
            if "image" not in content_type:
                LOGGER.warning(
                    "Skipping %s due to invalid content-type %s",
                    items[index].url,
                    content_type,
                )
                continue
            downloaded.append(
                _DownloadedImage(position=index, item=items[index], data=result.content)
            )
        return downloaded

    def _compose_collage(self, images: Iterable[_DownloadedImage]) -> bytes:
        sources: list[_PastedImage] = []
        for downloaded in images:
            image = Image.open(io.BytesIO(downloaded.data))
            converted = image.convert("RGBA")
            sources.append(_PastedImage(image=converted, display_index=downloaded.item.display_index))
            image.close()
        if not sources:
            raise ValueError("No images to compose")

        resized = self._normalize_heights(sources)
        collage, padding, placements = self._place_images(resized)
        if self._draw_divider and len(placements) > 1:
            self._add_dividers(collage, placements, padding)
        if self._draw_badges and placements:
            self._add_badges(collage, placements)
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
            normalized.append(_PastedImage(image=resized, display_index=item.display_index))
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
                scaled_images.append(_PastedImage(image=scaled, display_index=item.display_index))
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
                    display_index=item.display_index,
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

    def _add_badges(
        self,
        collage: Image.Image,
        placements: list[_Placement],
    ) -> None:
        draw = ImageDraw.Draw(collage, "RGBA")
        for placement in placements:
            x, y = self._badge_position(placement)
            bbox = [
                x,
                y,
                x + self._index_size_px,
                y + self._index_size_px,
            ]
            draw.ellipse(
                bbox,
                fill=self._index_bg_rgba,
                outline=self._index_stroke_rgba,
                width=max(1, self._index_stroke),
            )
            if self._badge_font is None:
                continue
            label = str(placement.display_index)
            text_width, text_height = self._measure_text(label)
            text_x = x + (self._index_size_px - text_width) / 2
            text_y = y + (self._index_size_px - text_height) / 2
            draw.text((text_x, text_y), label, font=self._badge_font, fill=self._index_text_rgba)

    def _badge_position(self, placement: _Placement) -> tuple[int, int]:
        x = placement.left + self._index_pad_px
        y = placement.top + self._index_pad_px
        x = max(placement.left, min(x, placement.right - self._index_size_px))
        y = max(placement.top, min(y, placement.bottom - self._index_size_px))
        return x, y

    def _measure_text(self, text: str) -> tuple[int, int]:
        font = self._badge_font
        if font is None:
            return (0, 0)
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            if bbox:
                left, top, right, bottom = bbox
                return right - left, bottom - top
        if hasattr(font, "getsize"):
            width, height = font.getsize(text)
            return width, height
        return (0, 0)

    def _prepare_badge_font(self) -> None:
        if not self._draw_badges:
            return
        font = self._load_numeric_font(self._index_text_size)
        self._badge_font = font

    def _load_numeric_font(
        self, size: int
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
        candidates = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
        for candidate in candidates:
            path = self._find_font_path(candidate)
            if path is None:
                continue
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None
        if not CollageService._font_warning_emitted:
            LOGGER.warning(
                "DejaVuSans fonts not available; using default Pillow font for collage badges",
            )
            CollageService._font_warning_emitted = True
        return font

    def _find_font_path(self, filename: str) -> Path | None:
        for base in PIL.__path__:
            base_path = Path(base)
            for folder in (base_path, base_path / "fonts", base_path / "Fonts"):
                candidate = folder / filename
                if candidate.exists():
                    return candidate
        return None

    @staticmethod
    def _hex_to_rgba(value: str, alpha: float) -> tuple[int, int, int, int]:
        value = value.lstrip("#")
        if len(value) not in {6, 3}:
            return (0, 0, 0, int(255 * alpha))
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        a = max(0, min(255, int(round(alpha * 255))))
        return r, g, b, a

