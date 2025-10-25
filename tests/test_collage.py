import asyncio
import io

import httpx
from PIL import Image

from app.services.collage import CollageItem, CollageResult, CollageService


def _make_image_bytes(size: tuple[int, int], color: str) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


async def _build_collage(
    transport: httpx.MockTransport,
    items: list[CollageItem],
    *,
    draw_divider: bool = True,
    draw_badges: bool = True,
) -> CollageResult | None:
    async with httpx.AsyncClient(transport=transport) as client:
        CollageService._font_warning_emitted = False
        service = CollageService(
            enabled=True,
            max_width=600,
            padding_px=20,
            cache_ttl_sec=300,
            draw_divider=draw_divider,
            draw_badges=draw_badges,
            index_size_px=48,
            index_pad_px=12,
            index_text_size=28,
            client=client,
        )
        return await service.build_collage(items)


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
            [
                CollageItem("https://example.com/img1.jpg", 1),
                CollageItem("https://example.com/img2.jpg", 2),
            ],
        )
    )

    assert result is not None
    assert result.included_positions == (0, 1)
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
            [
                CollageItem("https://example.com/img1.jpg", 1),
                CollageItem("https://example.com/img2.jpg", 2),
            ],
        )
    )

    assert result is not None
    assert result.included_positions == (0,)
    with Image.open(io.BytesIO(result.image_bytes)) as collage:
        assert collage.width > 0
        assert collage.height > 0


def test_collage_draws_numeric_badges_and_divider() -> None:
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
            [
                CollageItem("https://example.com/one.jpg", 1),
                CollageItem("https://example.com/two.jpg", 2),
            ],
            draw_divider=True,
            draw_badges=True,
        )
    )

    assert result is not None
    with Image.open(io.BytesIO(result.image_bytes)) as collage:
        width, height = collage.size
        line_x = 20 + 200 + 10
        middle_y = height // 2
        divider_pixel = collage.getpixel((line_x, middle_y))
        assert max(divider_pixel) < 250
        badge_x = 20 + 12 + 5
        badge_y = 20 + 12 + 5
        badge_pixel = collage.getpixel((badge_x, badge_y))
        assert badge_pixel != (255, 255, 255)


def test_collage_badges_fallback_to_default_font(monkeypatch) -> None:
    image_one = _make_image_bytes((180, 180), "#CCCCCC")
    image_two = _make_image_bytes((180, 180), "#999999")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("one.png"):
            return httpx.Response(200, content=image_one, headers={"content-type": "image/png"})
        if request.url.path.endswith("two.png"):
            return httpx.Response(200, content=image_two, headers={"content-type": "image/png"})
        return httpx.Response(404)

    monkeypatch.setattr(CollageService, "_find_font_path", lambda self, filename: None)

    transport = httpx.MockTransport(handler)
    result = asyncio.run(
        _build_collage(
            transport,
            [
                CollageItem("https://example.com/one.png", 1),
                CollageItem("https://example.com/two.png", 2),
            ],
        )
    )

    assert result is not None
    assert CollageService._font_warning_emitted
