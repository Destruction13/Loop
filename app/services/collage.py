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
    """Build a dual-portrait collage with optional padding and separator."""

    columns = 2
    padded_sources = list(image_urls[:columns])
    if len(padded_sources) < columns:
        padded_sources.extend([None] * (columns - len(padded_sources)))
    if not any(padded_sources):
        raise CollageSourceUnavailable("No image URLs provided")

    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(10.0, connect=10.0, read=10.0)
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    downloaded: list[_CollageSource] | None = None
    try:
        downloaded = await _download_sources(client, padded_sources)
        if not any(source.data for source in downloaded):
            raise CollageSourceUnavailable("Failed to download collage images")
        return await asyncio.to_thread(_compose_collage, downloaded, cfg)
    except CollageSourceUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "Collage pipeline failed", extra={"collage_sources": padded_sources}
        )
        raise CollageProcessingError(str(exc)) from exc
    finally:
        if owns_client:
            await client.aclose()


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
    slot_width = max(cfg.slot_width, 1)
    slot_height = max(cfg.slot_height, 1)
    separator_width = max(cfg.separator_width, 0)
    padding = max(cfg.padding, 0)

    canvas_width = slot_width * 2 + separator_width + padding * 2
    canvas_height = slot_height + padding * 2

    fmt = (cfg.output_format or "PNG").upper()
    raw_background = (cfg.background or "").strip()
    background_is_transparent = (
        fmt == "PNG" and (not raw_background or raw_background.lower() == "transparent")
    )

    if background_is_transparent:
        canvas_mode = "RGBA"
        background_color = (0, 0, 0, 0)
    else:
        canvas_mode = "RGB"
        background_color = _parse_color(
            raw_background or None,
            mode="RGB",
            fallback=(255, 255, 255),
        )

    canvas = Image.new(canvas_mode, (canvas_width, canvas_height), color=background_color)
    draw = ImageDraw.Draw(canvas)

    separator_color = _parse_color(
        cfg.separator_color,
        mode="RGBA" if canvas_mode == "RGBA" else "RGB",
        fallback=(42, 42, 42, 255) if canvas_mode == "RGBA" else (42, 42, 42),
    )

    tile_boxes = [
        (padding, padding, padding + slot_width, padding + slot_height),
        (
            padding + slot_width + separator_width,
            padding,
            padding + slot_width * 2 + separator_width,
            padding + slot_height,
        ),
    ]

    if separator_width > 0:
        separator_left = padding + slot_width
        separator_box = (
            separator_left,
            padding,
            separator_left + separator_width,
            padding + slot_height,
        )
        draw.rectangle(separator_box, fill=separator_color)

    target_size = (slot_width, slot_height)
    for index, source in enumerate(sources):
        if index >= len(tile_boxes):
            break
        image = _open_source_image(source)
        if image is None:
            continue
        try:
            fitted = ImageOps.fit(
                image,
                target_size,
                method=Image.LANCZOS,
                centering=(0.5, 0.5),
            )
            if fitted.size != target_size:
                fitted = fitted.resize(target_size, Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fit collage image %s: %s", source.url, exc)
            image.close()
            continue

        paste_left, paste_top, _, _ = tile_boxes[index]
        paste_position = (paste_left, paste_top)
        mask = fitted if fitted.mode in {"LA", "RGBA", "PA"} else None
        try:
            canvas.paste(fitted, paste_position, mask)
        finally:
            fitted.close()
            image.close()

    buffer = io.BytesIO()
    fmt = (cfg.output_format or "PNG").upper()
    save_kwargs: dict[str, object] = {"format": fmt}
    image_to_save = canvas
    if fmt == "JPEG":
        quality = max(min(cfg.jpeg_quality, 100), 1)
        save_kwargs.update({"quality": quality, "optimize": True, "progressive": True})
        image_to_save = canvas.convert("RGB")

    try:
        image_to_save.save(buffer, **save_kwargs)
    finally:
        if image_to_save is not canvas:
            image_to_save.close()
        canvas.close()

    buffer.seek(0)
    return buffer


def _open_source_image(source: _CollageSource) -> Image.Image | None:
    if source.data is None:
        return None

    try:
        with Image.open(io.BytesIO(source.data)) as original:
            return original.convert("RGBA")
    except UnidentifiedImageError:
        LOGGER.warning("Collage image %s is not a valid image", source.url)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to prepare collage image %s: %s", source.url, exc)
    return None


def _parse_color(
    value: str | None,
    *,
    mode: str,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    if value:
        normalized = value.strip()
        if normalized:
            if normalized.lower() == "transparent" and mode == "RGBA":
                return (0, 0, 0, 0)
            try:
                return ImageColor.getcolor(normalized, mode)
            except ValueError:
                LOGGER.warning(
                    "Unsupported color value %s, falling back to default", value
                )
    return fallback
