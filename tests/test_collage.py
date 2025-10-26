from __future__ import annotations

import asyncio
import io

import httpx
import pytest
from PIL import Image, ImageColor

from app.config import CollageConfig
from app.services.collage import CollageSourceUnavailable, build_three_tile_collage


def _default_cfg() -> CollageConfig:
    return CollageConfig(
        width=1600,
        height=800,
        columns=2,
        margin=30,
        divider_width=6,
        divider_color="#E5E5E5",
        background="#FFFFFF",
        jpeg_quality=88,
    )


def _tile_boxes(cfg: CollageConfig) -> list[tuple[int, int, int, int]]:
    usable_width = cfg.width - 2 * cfg.margin - cfg.divider_width * (cfg.columns - 1)
    tile_height = cfg.height - 2 * cfg.margin
    tile_width = usable_width / cfg.columns
    boxes: list[tuple[int, int, int, int]] = []
    current_left = cfg.margin
    for index in range(cfg.columns):
        left = int(round(current_left))
        if index == cfg.columns - 1:
            right = cfg.width - cfg.margin
        else:
            right = int(round(current_left + tile_width))
        if right <= left:
            right = left + 1
        boxes.append((left, cfg.margin, right, cfg.margin + tile_height))
        current_left = right + cfg.divider_width
    return boxes


def _make_image_bytes(size: tuple[int, int], color: str) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


async def _render_collage(
    transport: httpx.MockTransport,
    urls: list[str | None],
    cfg: CollageConfig,
):
    async with httpx.AsyncClient(transport=transport) as client:
        return await build_three_tile_collage(urls, cfg, client=client)


def test_collage_1x2_geometry() -> None:
    config = _default_cfg()
    red = _make_image_bytes((400, 400), "red")
    green = _make_image_bytes((400, 400), "green")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("a.jpg"):
            return httpx.Response(200, content=red, headers={"content-type": "image/jpeg"})
        if request.url.path.endswith("b.jpg"):
            return httpx.Response(200, content=green, headers={"content-type": "image/jpeg"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            [
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
            ],
            config,
        )
    )

    with Image.open(buffer) as collage:
        assert collage.size == (config.width, config.height)
        boxes = _tile_boxes(config)
        centers = [((left + right) // 2, (top + bottom) // 2) for left, top, right, bottom in boxes]
        first_pixel = collage.getpixel(centers[0])
        second_pixel = collage.getpixel(centers[1])
        assert first_pixel[0] > 150 and first_pixel[1] < 100 and first_pixel[2] < 100
        assert second_pixel[1] > 120 and second_pixel[0] < 120


def test_collage_draws_dividers() -> None:
    config = _default_cfg()
    config.divider_width = 8
    config.divider_color = "#123456"
    image_bytes = _make_image_bytes((500, 500), "white")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=image_bytes,
            headers={"content-type": "image/jpeg"},
        )

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            ["https://example.com/1.jpg", "https://example.com/2.jpg"],
            config,
        )
    )

    divider_rgb = ImageColor.getrgb(config.divider_color)
    with Image.open(buffer) as collage:
        boxes = _tile_boxes(config)
        divider_left = boxes[0][2]
        sample_x = divider_left + config.divider_width // 2
        sample = collage.getpixel((sample_x, config.height // 2))
        assert all(abs(channel - ref) <= 5 for channel, ref in zip(sample, divider_rgb))


def test_collage_handles_missing_sources() -> None:
    config = _default_cfg()
    red = _make_image_bytes((400, 400), "red")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=red, headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            ["https://example.com/a.jpg", None],
            config,
        )
    )

    with Image.open(buffer) as collage:
        boxes = _tile_boxes(config)
        background_rgb = ImageColor.getrgb(config.background)
        left_pixel = collage.getpixel(((boxes[0][0] + boxes[0][2]) // 2, (boxes[0][1] + boxes[0][3]) // 2))
        right_pixel = collage.getpixel(((boxes[1][0] + boxes[1][2]) // 2, (boxes[1][1] + boxes[1][3]) // 2))
        assert left_pixel[0] > 150
        assert right_pixel == background_rgb


def test_collage_raises_when_all_sources_missing() -> None:
    config = _default_cfg()

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - no calls
        raise AssertionError("Handler should not be called")

    transport = httpx.MockTransport(handler)

    with pytest.raises(CollageSourceUnavailable):
        asyncio.run(
            _render_collage(
                transport,
                [None, None],
                config,
            )
        )
