"""Utilities for building three-tile collages used in model recommendations."""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx
from PIL import Image, ImageColor, ImageDraw, ImageOps, UnidentifiedImageError

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


async def build_three_tile_collage(
    image_urls: Sequence[str | None],
    cfg: CollageConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> io.BytesIO:
    """Build a 1×N collage (default 1×3) with vertical dividers between tiles."""

    columns = max(cfg.columns, 1)
    padded_sources = list(image_urls[:columns])
    if len(padded_sources) < columns:
        padded_sources.extend([None] * (columns - len(padded_sources)))
    if not any(padded_sources):
        raise CollageSourceUnavailable("No image URLs provided")

    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(10.0, connect=10.0, read=10.0)
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        downloaded = await _download_sources(client, padded_sources)
    finally:
        if owns_client:
            await client.aclose()

    if not any(source.data for source in downloaded):
        raise CollageSourceUnavailable("Failed to download collage images")

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
    divider_color = ImageColor.getrgb(cfg.divider_color)
    width = max(cfg.width, 1)
    height = max(cfg.height, 1)
    margin = max(cfg.margin, 0)
    columns = max(cfg.columns, 1)
    divider_width = max(cfg.divider_width, 0)

    usable_width = max(width - 2 * margin - divider_width * (columns - 1), 1)
    tile_height = max(height - 2 * margin, 1)
    tile_width_float = usable_width / columns

    canvas = Image.new("RGB", (width, height), color=background)
    draw = ImageDraw.Draw(canvas)

    tile_boxes: list[tuple[int, int, int, int]] = []
    current_left = margin
    for index in range(columns):
        left = int(round(current_left))
        if index == columns - 1:
            right = width - margin
        else:
            right = int(round(current_left + tile_width_float))
        if right <= left:
            right = left + 1
        tile_boxes.append((left, margin, right, margin + tile_height))
        current_left = right + divider_width

    for index, source in enumerate(sources):
        if index >= len(tile_boxes):
            break
        image = _load_source_image(source, tile_boxes[index], background)
        if image is None:
            continue
        tile_left, tile_top, tile_right, tile_bottom = tile_boxes[index]
        cell_width = tile_right - tile_left
        cell_height = tile_bottom - tile_top
        pos_x = tile_left + max((cell_width - image.width) // 2, 0)
        pos_y = tile_top + max((cell_height - image.height) // 2, 0)
        canvas.paste(image, (pos_x, pos_y))
        image.close()

    if divider_width > 0:
        for index in range(1, columns):
            divider_left = tile_boxes[index - 1][2]
            divider_right = divider_left + divider_width
            divider_box = (divider_left, margin, divider_right, margin + tile_height)
            draw.rectangle(divider_box, fill=divider_color)

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
    cell_box: tuple[int, int, int, int],
    background: tuple[int, int, int],
) -> Image.Image | None:
    if source.data is None:
        return None

    cell_width = max(cell_box[2] - cell_box[0], 1)
    cell_height = max(cell_box[3] - cell_box[1], 1)

    try:
        with Image.open(io.BytesIO(source.data)) as original:
            converted = original.convert("RGB")
            fitted = ImageOps.contain(
                converted, (cell_width, cell_height), method=Image.LANCZOS
            )
            if fitted is not converted:
                converted.close()
            return fitted
    except UnidentifiedImageError:
        LOGGER.warning("Collage image %s is not a valid image", source.url)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to prepare collage image %s: %s", source.url, exc)
    return None
