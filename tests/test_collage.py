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
        slot_width=1080,
        slot_height=1440,
        separator_width=24,
        padding=0,
        separator_color="#2A2A2A",
        background="#000000",
        output_format="PNG",
        jpeg_quality=90,
    )


def _geometry(cfg: CollageConfig) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    width = cfg.width
    height = cfg.height
    left_a = cfg.padding
    top = cfg.padding
    left_b = cfg.padding + cfg.slot_width + cfg.separator_width
    boxes = [
        (left_a, top, left_a + cfg.slot_width, top + cfg.slot_height),
        (left_b, top, left_b + cfg.slot_width, top + cfg.slot_height),
    ]
    return (width, height), boxes


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


def test_collage_dual_portrait_geometry() -> None:
    config = _default_cfg()
    config.padding = 48
    config.background = "#F7F7F7"
    config.separator_color = "#FFFFFF"
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
        expected_size, boxes = _geometry(config)
        assert collage.size == expected_size
        centers = [
            ((left + right) // 2, (top + bottom) // 2)
            for left, top, right, bottom in boxes
        ]
        first_pixel = collage.getpixel(centers[0])
        second_pixel = collage.getpixel(centers[1])
        assert first_pixel[0] > 150 and first_pixel[1] < 100 and first_pixel[2] < 100
        assert second_pixel[1] > 120 and second_pixel[0] < 120


def test_collage_draws_dividers() -> None:
    config = _default_cfg()
    config.separator_width = 18
    config.separator_color = "#CCCCCC"
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

    expected_color = (204, 204, 204, 255)
    with Image.open(buffer) as collage:
        _, boxes = _geometry(config)
        divider_left = boxes[0][2]
        sample_x = divider_left + config.separator_width // 2
        sample = collage.getpixel((sample_x, config.slot_height // 2 + config.padding))
        if len(sample) == 3:
            sample = (*sample, 255)
        assert all(abs(channel - ref) <= 5 for channel, ref in zip(sample, expected_color))


def test_collage_handles_missing_sources() -> None:
    config = _default_cfg()
    config.padding = 48
    config.background = "#222222"
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
        _, boxes = _geometry(config)
        background_rgb = ImageColor.getcolor(config.background, "RGB")
        left_pixel = collage.getpixel(
            ((boxes[0][0] + boxes[0][2]) // 2, (boxes[0][1] + boxes[0][3]) // 2)
        )
        right_pixel = collage.getpixel(
            ((boxes[1][0] + boxes[1][2]) // 2, (boxes[1][1] + boxes[1][3]) // 2)
        )
        assert left_pixel[0] > 150
        assert tuple(right_pixel[:3]) == background_rgb


@pytest.mark.parametrize("background_value", ["", "transparent"])
def test_collage_supports_transparent_background(background_value: str) -> None:
    config = _default_cfg()
    config.background = background_value
    red = _make_image_bytes((600, 900), "red")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=red,
            headers={"content-type": "image/jpeg"},
        )

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            ["https://example.com/a.jpg", None],
            config,
        )
    )

    with Image.open(buffer) as collage:
        assert collage.mode == "RGBA"
        bg_x = config.slot_width + config.separator_width + config.padding + 1
        bg_y = config.padding + 1
        top_right = collage.getpixel((bg_x, bg_y))
        assert top_right[3] == 0


def test_collage_can_export_jpeg() -> None:
    config = _default_cfg()
    config.output_format = "JPEG"
    config.jpeg_quality = 85
    blue = _make_image_bytes((700, 700), "blue")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=blue,
            headers={"content-type": "image/jpeg"},
        )

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            ["https://example.com/a.jpg", "https://example.com/b.jpg"],
            config,
        )
    )

    with Image.open(buffer) as collage:
        assert collage.format == "JPEG"
        assert collage.mode == "RGB"


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


def test_collage_config_width_height_properties() -> None:
    cfg = _default_cfg()
    cfg.padding = 12
    cfg.separator_width = 8
    assert cfg.width == cfg.slot_width * 2 + cfg.separator_width + cfg.padding * 2
    assert cfg.height == cfg.slot_height + cfg.padding * 2


def test_collage_invalid_background_falls_back_to_white() -> None:
    config = _default_cfg()
    config.background = "not-a-color"
    config.separator_width = 0
    red = _make_image_bytes((500, 500), "red")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=red,
            headers={"content-type": "image/jpeg"},
        )

    transport = httpx.MockTransport(handler)
    buffer = asyncio.run(
        _render_collage(
            transport,
            [None, "https://example.com/b.jpg"],
            config,
        )
    )

    with Image.open(buffer) as collage:
        assert collage.getpixel((0, 0))[0:3] == (255, 255, 255)
