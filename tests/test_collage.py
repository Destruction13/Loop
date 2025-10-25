"""Tests for the collage service."""

from __future__ import annotations

import asyncio
import io

import httpx
from PIL import Image

from app.services.collage import CollageResult, CollageService


def _make_image_bytes(size: tuple[int, int], color: str) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


async def _build_collage(
    transport: httpx.MockTransport,
    urls: list[str],
    *,
    draw_divider: bool = True,
    draw_badges: bool = True,
    badge_style: str = "circle",
) -> CollageResult | None:
    async with httpx.AsyncClient(transport=transport) as client:
        service = CollageService(
            enabled=True,
            max_width=600,
            padding_px=20,
            cache_ttl_sec=300,
            draw_divider=draw_divider,
            draw_badges=draw_badges,
            badge_style=badge_style,
            client=client,
        )
        return await service.build_collage(urls)


def test_collage_combines_two_images() -> None:
    image_one = _make_image_bytes((400, 300), "red")
    image_two = _make_image_bytes((200, 500), "blue")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("img1.jpg"):
            return httpx.Response(200, content=image_one, headers={"content-type": "image/jpeg"})
        if request.url.path.endswith("img2.jpg"):
            return httpx.Response(200, content=image_two, headers={"content-type": "image/jpeg"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    result = asyncio.run(
        _build_collage(
            transport,
            ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
        )
    )

    assert result is not None
    assert result.included_indices == (0, 1)
    with Image.open(io.BytesIO(result.image_bytes)) as collage:
        assert collage.width <= 600
        assert collage.height > 0


def test_collage_handles_partial_download() -> None:
    image_one = _make_image_bytes((300, 300), "green")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("img1.jpg"):
            return httpx.Response(200, content=image_one, headers={"content-type": "image/jpeg"})
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    result = asyncio.run(
        _build_collage(
            transport,
            ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
        )
    )

    assert result is not None
    assert result.included_indices == (0,)
    with Image.open(io.BytesIO(result.image_bytes)) as collage:
        assert collage.width > 0
        assert collage.height > 0


def test_collage_draws_divider_and_badges() -> None:
    image_one = _make_image_bytes((200, 200), "red")
    image_two = _make_image_bytes((200, 200), "green")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("one.jpg"):
            return httpx.Response(200, content=image_one, headers={"content-type": "image/jpeg"})
        if request.url.path.endswith("two.jpg"):
            return httpx.Response(200, content=image_two, headers={"content-type": "image/jpeg"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    result = asyncio.run(
        _build_collage(
            transport,
            ["https://example.com/one.jpg", "https://example.com/two.jpg"],
        )
    )

    assert result is not None
    with Image.open(io.BytesIO(result.image_bytes)) as collage:
        width, height = collage.size
        line_x = 20 + 200 + 10
        middle_y = height // 2
        divider_pixel = collage.getpixel((line_x, middle_y))
        assert max(divider_pixel) < 250
        badge_x = 20 + 12 + 10
        badge_y = 20 + 12 + 10
        badge_pixel = collage.getpixel((badge_x, badge_y))
        assert badge_pixel != (255, 255, 255)
