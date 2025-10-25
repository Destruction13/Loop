"""Utilities for building collage previews of catalog models."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger("loop_bot.collage")


@dataclass(frozen=True)
class CollageResult:
    """Result of a collage generation request."""

    image_bytes: bytes
    included_indices: tuple[int, ...]


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
        draw_badges: bool = True,
        badge_style: str = "circle",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._enabled = enabled
        self._max_width = max_width
        self._padding_px = padding_px
        self._cache_ttl = cache_ttl_sec
        self._draw_divider = draw_divider
        self._draw_badges = draw_badges
        self._badge_style = badge_style
        if client is None:
            timeout = httpx.Timeout(10.0, connect=10.0, read=10.0)
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
            self._client_owner = True
        else:
            self._client = client
            self._client_owner = False
        self._cache: dict[tuple[str, ...], tuple[float, CollageResult]] = {}
        self._cache_lock = asyncio.Lock()
        self._badge_font = self._load_font()
        self._font_supports_circled = self._detect_circled_support(self._badge_font)
        self._circled_digits = {
            1: "①",
            2: "②",
            3: "③",
            4: "④",
        }

    @property
    def enabled(self) -> bool:
        """Return whether collage generation is enabled."""

        return self._enabled

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned."""

        if self._client_owner:
            await self._client.aclose()

    async def build_collage(self, urls: Sequence[str]) -> CollageResult | None:
        """Build a collage for the given image URLs.

        Returns ``None`` when collage generation is disabled or failed for all
        images. The order of ``urls`` is preserved in the resulting indices.
        """

        if not self._enabled:
            return None

        normalized = tuple(urls)
        cached = await self._get_from_cache(normalized)
        if cached is not None:
            return cached

        downloaded = await self._download_images(normalized)
        if not downloaded:
            return None

        indices = tuple(index for index, _ in downloaded)
        try:
            image_bytes = await asyncio.to_thread(
                self._compose_collage, (data for _, data in downloaded)
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to compose collage for %s: %s", normalized, exc)
            return None

        result = CollageResult(image_bytes=image_bytes, included_indices=indices)
        if len(indices) == len(normalized):
            await self._store_in_cache(normalized, result)
        return result

    async def _get_from_cache(self, key: tuple[str, ...]) -> CollageResult | None:
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
        self, key: tuple[str, ...], result: CollageResult
    ) -> None:
        async with self._cache_lock:
            self._cache[key] = (time.monotonic() + self._cache_ttl, result)

    async def _download_images(
        self, urls: Sequence[str]
    ) -> list[tuple[int, bytes]]:
        tasks = [self._client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        downloaded: list[tuple[int, bytes]] = []
        for index, result in enumerate(responses):
            if isinstance(result, Exception):
                LOGGER.warning("Failed to download %s: %s", urls[index], result)
                continue
            if result.status_code >= 400:
                LOGGER.warning(
                    "Image request failed for %s with status %s",
                    urls[index],
                    result.status_code,
                )
                continue
            content_type = result.headers.get("content-type", "")
            if "image" not in content_type:
                LOGGER.warning(
                    "Skipping %s due to invalid content-type %s",
                    urls[index],
                    content_type,
                )
                continue
            downloaded.append((index, result.content))
        return downloaded

    def _compose_collage(self, images: Iterable[bytes]) -> bytes:
        pil_images: list[Image.Image] = []
        for data in images:
            image = Image.open(io.BytesIO(data))
            converted = image.convert("RGB")
            pil_images.append(converted)
            image.close()
        if not pil_images:
            raise ValueError("No images to compose")

        resized = self._normalize_heights(pil_images)
        collage, padding, placements = self._place_images(resized)
        if self._draw_divider and len(placements) > 1:
            self._add_dividers(collage, placements, padding)
        if self._draw_badges and placements:
            self._add_badges(collage, placements)
        buffer = io.BytesIO()
        try:
            collage.save(buffer, format="JPEG", quality=85)
        finally:
            collage.close()
            seen: set[int] = set()
            for image in resized:
                identifier = id(image)
                if identifier in seen:
                    continue
                image.close()
                seen.add(identifier)
            for image in pil_images:
                identifier = id(image)
                if identifier in seen:
                    continue
                image.close()
        return buffer.getvalue()

    def _normalize_heights(self, images: list[Image.Image]) -> list[Image.Image]:
        target_height = max(image.height for image in images)
        normalized: list[Image.Image] = []
        for image in images:
            if image.height == target_height:
                normalized.append(image)
                continue
            ratio = target_height / image.height
            new_width = max(1, int(image.width * ratio))
            resized = image.resize((new_width, target_height), Image.LANCZOS)
            normalized.append(resized)
        return normalized

    def _place_images(
        self, images: list[Image.Image]
    ) -> tuple[Image.Image, int, list[tuple[int, int, int, int]]]:
        padding = self._padding_px
        total_width = sum(image.width for image in images) + padding * (len(images) + 1)
        height = images[0].height if images else 0

        if total_width > self._max_width:
            scale = self._max_width / total_width
            padding = max(1, int(padding * scale))
            scaled_images: list[Image.Image] = []
            for image in images:
                new_width = max(1, int(image.width * scale))
                new_height = max(1, int(image.height * scale))
                scaled_images.append(image.resize((new_width, new_height), Image.LANCZOS))
            images = scaled_images
            height = images[0].height
            total_width = sum(image.width for image in images) + padding * (len(images) + 1)
        else:
            height = images[0].height

        canvas_height = height + padding * 2
        collage = Image.new("RGB", (total_width, canvas_height), color="#FFFFFF")
        current_x = padding
        placements: list[tuple[int, int, int, int]] = []
        for image in images:
            y = padding + (height - image.height) // 2
            collage.paste(image, (current_x, y))
            placements.append((current_x, y, current_x + image.width, y + image.height))
            current_x += image.width + padding
        return collage, padding, placements

    def _add_dividers(
        self,
        collage: Image.Image,
        placements: list[tuple[int, int, int, int]],
        padding: int,
    ) -> None:
        divider_color = (229, 229, 229)
        divider_width = 2
        draw = ImageDraw.Draw(collage)
        top = padding
        bottom = collage.height - padding
        for left_box, right_box in zip(placements, placements[1:]):
            gap_start = left_box[2]
            gap_end = right_box[0]
            if gap_end <= gap_start:
                continue
            x = gap_start + (gap_end - gap_start) // 2
            draw.line([(x, top), (x, bottom)], fill=divider_color, width=divider_width)

    def _add_badges(
        self,
        collage: Image.Image,
        placements: list[tuple[int, int, int, int]],
    ) -> None:
        draw = ImageDraw.Draw(collage, "RGBA")
        diameter = 40
        margin = 12
        for index, (left, top, right, bottom) in enumerate(placements, start=1):
            label = self._badge_label(index)
            bbox_left = left + margin
            bbox_top = top + margin
            bbox_right = min(right - margin, bbox_left + diameter)
            bbox_bottom = min(bottom - margin, bbox_top + diameter)
            if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
                continue
            if self._badge_style == "circle":
                draw.ellipse(
                    [bbox_left, bbox_top, bbox_right, bbox_bottom],
                    fill=(0, 0, 0, 128),
                )
                text_color = (255, 255, 255, 255)
            else:
                text_color = (0, 0, 0, 255)
            text_width, text_height = self._measure_text(label)
            text_x = bbox_left + (bbox_right - bbox_left - text_width) / 2
            text_y = bbox_top + (bbox_bottom - bbox_top - text_height) / 2
            draw.text((text_x, text_y), label, font=self._badge_font, fill=text_color)

    def _badge_label(self, index: int) -> str:
        label = self._circled_digits.get(index, str(index))
        if not self._font_supports_circled:
            return str(index)
        return label

    def _measure_text(self, text: str) -> tuple[int, int]:
        bbox = self._badge_font.getbbox(text)
        if bbox is None:
            return (0, 0)
        left, top, right, bottom = bbox
        return (right - left, bottom - top)

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=22)
        except Exception:  # noqa: BLE001
            return ImageFont.load_default()

    def _detect_circled_support(self, font: ImageFont.ImageFont) -> bool:
        try:
            bbox = font.getbbox("①")
        except Exception:  # noqa: BLE001
            return False
        if not bbox:
            return False
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width > 0 and height > 0

