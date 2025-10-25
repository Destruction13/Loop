"""Utilities for building two-up collages used in model recommendations."""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx
from PIL import (
    Image,
    ImageColor,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageOps,
    UnidentifiedImageError,
)

from app.config import CollageConfig

LOGGER = logging.getLogger("loop_bot.collage")


class CollageSourceUnavailable(Exception):
    """Raised when no valid sources are available to compose a collage."""


class CollageProcessingError(Exception):
    """Raised when Pillow fails to compose a collage from the inputs."""


@dataclass(slots=True)
class _CollageSource:
    """Container for a single collage source download."""

    url: str | None
    data: bytes | None


async def build_two_up_collage(
    left_url: str | None,
    right_url: str | None,
    cfg: CollageConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> io.BytesIO:
    """Build a two-image collage according to the provided configuration.

    The function downloads both images, fills missing slots with placeholders and
    composes them into a single JPEG. When both images are unavailable the
    ``CollageSourceUnavailable`` exception is raised. Fatal Pillow errors during
    composition are surfaced as ``CollageProcessingError``.
    """

    sources = [left_url, right_url]
    if not any(sources):
        raise CollageSourceUnavailable("No image URLs provided")

    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(10.0, connect=10.0, read=10.0)
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        downloaded = await _download_sources(client, sources)
    finally:
        if owns_client:
            await client.aclose()

    if not any(source.data for source in downloaded):
        raise CollageSourceUnavailable("Failed to download both collage images")

    try:
        buffer = await asyncio.to_thread(_compose_collage, downloaded, cfg)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to compose collage: %s", exc)
        raise CollageProcessingError(str(exc)) from exc
    return buffer


async def _download_sources(
    client: httpx.AsyncClient, urls: Sequence[str | None]
) -> list[_CollageSource]:
    tasks = [client.get(url) if url else None for url in urls]
    results: list[_CollageSource] = []
    for index, task in enumerate(tasks):
        url = urls[index]
        if task is None or url is None:
            results.append(_CollageSource(url=url, data=None))
            continue
        try:
            response = await task
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            LOGGER.warning("Failed to download collage image %s: %s", url, exc)
            results.append(_CollageSource(url=url, data=None))
            continue
        status = response.status_code
        if status < 200 or status >= 300:
            LOGGER.warning(
                "Image request for %s returned status %s", url, status
            )
            results.append(_CollageSource(url=url, data=None))
            continue
        content_type = response.headers.get("content-type", "").lower()
        if "image" not in content_type:
            LOGGER.warning(
                "Skipping collage image %s due to content type %s", url, content_type
            )
            results.append(_CollageSource(url=url, data=None))
            continue
        results.append(_CollageSource(url=url, data=response.content))
    return results


def _compose_collage(sources: Iterable[_CollageSource], cfg: CollageConfig) -> io.BytesIO:
    background = ImageColor.getrgb(cfg.background)
    width = max(cfg.width, 1)
    height = max(cfg.height, 1)
    padding = max(cfg.padding, 0)
    gap = max(cfg.gap, 0)
    cell_width = max((width - 2 * padding - gap) // 2, 1)
    cell_height = max(height - 2 * padding, 1)

    canvas = Image.new("RGB", (width, height), color=background)

    for index, source in enumerate(sources):
        image = _load_source_image(source, (cell_width, cell_height), cfg, background)
        offset_x = padding + index * (cell_width + gap)
        pos_x = offset_x + max((cell_width - image.width) // 2, 0)
        pos_y = padding + max((cell_height - image.height) // 2, 0)
        canvas.paste(image, (pos_x, pos_y))
        image.close()

    sharpen_amount = max(0.0, min(cfg.sharpen, 1.0))
    if sharpen_amount > 0:
        percent = int(150 * sharpen_amount)
        canvas = canvas.filter(
            ImageFilter.UnsharpMask(radius=2, percent=max(percent, 1), threshold=3)
        )

    buffer = io.BytesIO()
    canvas.save(
        buffer,
        format="JPEG",
        quality=max(min(cfg.jpeg_quality, 100), 1),
        optimize=True,
        progressive=True,
    )
    buffer.seek(0)
    canvas.close()
    return buffer


def _load_source_image(
    source: _CollageSource,
    cell_size: tuple[int, int],
    cfg: CollageConfig,
    background: tuple[int, int, int],
) -> Image.Image:
    if source.data is None:
        return _build_placeholder(cell_size, background)

    try:
        with Image.open(io.BytesIO(source.data)) as original:
            converted = original.convert("RGB")
            fitted = _fit_image(converted, cell_size, cfg.fit_mode)
            if fitted is not converted:
                converted.close()
            return fitted
    except UnidentifiedImageError:
        LOGGER.warning("Collage image %s is not a valid image", source.url)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to prepare collage image %s: %s", source.url, exc)
    return _build_placeholder(cell_size, background)


def _fit_image(image: Image.Image, cell_size: tuple[int, int], fit_mode: str) -> Image.Image:
    target = tuple(max(value, 1) for value in cell_size)
    if fit_mode == "cover":
        return ImageOps.fit(image, target, method=Image.LANCZOS, centering=(0.5, 0.5))
    return ImageOps.contain(image, target, method=Image.LANCZOS)


def _build_placeholder(
    cell_size: tuple[int, int], background: tuple[int, int, int]
) -> Image.Image:
    width, height = cell_size
    border_color = (200, 200, 200)
    text_color = (160, 160, 160)

    image = Image.new("RGB", (width, height), color=background)
    draw = ImageDraw.Draw(image)
    border_width = max(min(width, height) // 30, 2)
    inner_box = (
        border_width // 2,
        border_width // 2,
        width - border_width // 2 - 1,
        height - border_width // 2 - 1,
    )
    draw.rectangle(inner_box, outline=border_color, width=border_width)

    text = "NO IMAGE"
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    if text_width < width and text_height < height:
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        draw.text((text_x, text_y), text, fill=text_color, font=font)

    return image
