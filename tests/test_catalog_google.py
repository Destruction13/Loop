"""Tests for Google Sheets catalog integration."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.services.catalog_base import CatalogError
from app.services.catalog_google import GoogleCatalogConfig, GoogleSheetCatalog
from app.utils.drive import drive_view_to_direct


def test_drive_view_to_direct_extracts_id() -> None:
    url = "https://drive.google.com/file/d/ABC123/view?usp=sharing"
    direct = drive_view_to_direct(url)
    assert direct == "https://drive.google.com/uc?export=view&id=ABC123"


def test_catalog_parses_and_filters_by_gender() -> None:
    csv_text = (
        "Название,Модель,Ссылка на сайт,Пол,Ссылка на изображение для пользователя,"
        "Ссылка на изображение для NanoBanana,Уникальный ID\n"
        "Alpha,A1,https://example.com/a,Мужской,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha\n"
        "Bravo,B1,https://example.com/b,Жен.,https://drive.google.com/file/d/BBB/view,,\n"
        "Charlie,C1,https://example.com/c,уни,https://drive.google.com/file/d/CCC/view,https://nano/c,\n"
        ",,https://example.com/d,Мужской,https://drive.google.com/file/d/DDD/view,,\n"
    )

    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=csv_text)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(csv_url="https://example.com", cache_ttl_seconds=60, retries=3),
                client=client,
            )
            male_models = await catalog.list_by_gender("male")
            assert [model.title for model in male_models] == ["Alpha", "Charlie"]
            assert male_models[0].img_user_url.endswith("id=AAA")
            assert male_models[1].gender == "Унисекс"
            female_models = await catalog.list_by_gender("female")
            assert [model.title for model in female_models] == ["Bravo", "Charlie"]
            seen_first = {female_models[0].unique_id}
            picks = await catalog.pick_four("female", seen_first)
            assert picks
            assert picks[0].unique_id != female_models[0].unique_id
            await catalog.aclose()

    asyncio.run(_run())


def test_retry_policy_handles_redirect_and_server_errors() -> None:
    csv_text = (
        "Название,Ссылка на сайт,Ссылка на изображение для пользователя\n"
        "Alpha,https://example.com/a,https://drive.google.com/file/d/AAA/view\n"
    )

    async def _run() -> None:
        counters = {"initial": 0, "final": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/initial":
                counters["initial"] += 1
                return httpx.Response(307, headers={"Location": "https://example.com/final"})
            if request.url.path == "/final":
                counters["final"] += 1
                if counters["final"] == 1:
                    return httpx.Response(500, text="error")
                return httpx.Response(200, text=csv_text)
            raise AssertionError(f"Unexpected path {request.url.path}")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url="https://example.com/initial",
                    cache_ttl_seconds=60,
                    retries=3,
                ),
                client=client,
            )
            models = await catalog.list_by_gender("male")
            assert models, "Expected models after retries"
            assert counters["initial"] == 2
            assert counters["final"] == 2
            await catalog.aclose()

    asyncio.run(_run())


def test_catalog_cache_respects_ttl() -> None:
    csv_text = (
        "Название,Ссылка на сайт,Ссылка на изображение для пользователя\n"
        "Alpha,https://example.com/a,https://drive.google.com/file/d/AAA/view\n"
    )

    async def _run() -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] > 1:
                raise AssertionError("Unexpected additional network call")
            return httpx.Response(200, text=csv_text)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url="https://example.com/cache",
                    cache_ttl_seconds=60,
                    retries=3,
                ),
                client=client,
            )
            await catalog.list_by_gender("male")
            await catalog.list_by_gender("male")
            assert calls["count"] == 1
            await catalog.aclose()

    asyncio.run(_run())


def test_html_response_raises_not_csv() -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<!doctype html><html></html>")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url="https://example.com/html",
                    cache_ttl_seconds=60,
                    retries=2,
                ),
                client=client,
            )
            with pytest.raises(CatalogError, match="Not CSV"):
                await catalog.list_by_gender("male")
            await catalog.aclose()

    asyncio.run(_run())
