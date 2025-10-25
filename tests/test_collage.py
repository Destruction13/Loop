from __future__ import annotations

import asyncio
import io

import httpx
import pytest
from PIL import Image, ImageColor

from app.config import CollageConfig
from app.services.collage import (
    CollageSourceUnavailable,
    build_two_up_collage,
)


def _make_image_bytes(size: tuple[int, int], color: str) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


async def _render_collage(
    transport: httpx.MockTransport,
    left_url: str | None,
    right_url: str | None,
    cfg: CollageConfig,
):
    async with httpx.AsyncClient(transport=transport) as client:
        return await build_two_up_collage(left_url, right_url, cfg, client=client)


def _default_cfg() -> CollageConfig:
    return CollageConfig(
        width=640,
        height=320,
        gap=20,
        padding=16,
        background="#FFFFFF",
        jpeg_quality=90,
        fit_mode="contain",
        sharpen=0.0,
        divider=6,
        divider_color="#E6E9EF",
        divider_radius=0,
        cell_border=0,
        cell_border_color="#D7DBE4",
    )


def test_collage_matches_config_dimensions() -> None:
    config = _default_cfg()
    left = _make_image_bytes((800, 400), "red")
    right = _make_image_bytes((400, 800), "blue")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("left.jpg"):
            return httpx.Response(200, content=left, headers={"content-type": "image/jpeg"})
        if request.url.path.endswith("right.jpg"):
            return httpx.Response(200, content=right, headers={"content-type": "image/jpeg"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            "https://example.com/left.jpg",
            "https://example.com/right.jpg",
            config,
        )
    )

    with Image.open(buffer) as collage:
        assert collage.size == (config.width, config.height)
        cell_width = (config.width - 2 * config.padding - config.gap) // 2
        left_center = collage.getpixel((config.padding + cell_width // 2, config.height // 2))
        right_center = collage.getpixel((config.padding + cell_width + config.gap + cell_width // 2, config.height // 2))
        assert left_center[0] > 150  # red channel dominant
        assert right_center[2] > 150  # blue channel dominant


def test_collage_draws_configured_divider() -> None:
    config = _default_cfg()
    config.jpeg_quality = 100
    config.divider = 12
    config.divider_color = "#123456"
    left = _make_image_bytes((400, 400), "white")
    right = _make_image_bytes((400, 400), "white")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=left if request.url.path.endswith("left.jpg") else right,
            headers={"content-type": "image/jpeg"},
        )

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            "https://example.com/left.jpg",
            "https://example.com/right.jpg",
            config,
        )
    )

    expected = ImageColor.getrgb(config.divider_color)
    with Image.open(buffer) as collage:
        cell_width = (config.width - 2 * config.padding - config.gap) // 2
        divider_left = config.padding + cell_width + max(
            (config.gap - config.divider) // 2,
            0,
        )
        sample_x = divider_left + config.divider // 2
        sample = collage.getpixel((sample_x, config.height // 2))
        assert all(abs(channel - ref) <= 5 for channel, ref in zip(sample, expected))


def test_collage_uses_placeholder_for_failed_download() -> None:
    config = _default_cfg()
    left = _make_image_bytes((500, 500), "green")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("left.jpg"):
            return httpx.Response(200, content=left, headers={"content-type": "image/jpeg"})
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            "https://example.com/left.jpg",
            "https://example.com/right.jpg",
            config,
        )
    )

    with Image.open(buffer) as collage:
        cell_width = (config.width - 2 * config.padding - config.gap) // 2
        placeholder_sample = collage.getpixel(
            (config.padding + cell_width + config.gap + 5, config.padding + 5)
        )
        assert all(channel >= 150 for channel in placeholder_sample)


def test_collage_raises_when_no_sources_available() -> None:
    config = _default_cfg()

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - no calls
        raise AssertionError("Handler should not be called")

    transport = httpx.MockTransport(handler)

    with pytest.raises(CollageSourceUnavailable):
        asyncio.run(
            _render_collage(
                transport,
                None,
                None,
                config,
            )
        )
