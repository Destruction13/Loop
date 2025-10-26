"""Tests for Google Sheets catalog integration."""

from __future__ import annotations

import asyncio
import logging

import random

import httpx
import pytest

from app.services.catalog_base import CatalogError
from app.services.catalog_google import GoogleCatalogConfig, GoogleSheetCatalog
from app.utils.drive import DriveFolderUrlError, DriveUrlError, drive_view_to_direct


@pytest.mark.parametrize(
    ("url", "expected_id"),
    [
        ("https://drive.google.com/file/d/ABC123/view?usp=sharing", "ABC123"),
        ("https://drive.google.com/open?id=XYZ789", "XYZ789"),
        ("https://drive.google.com/uc?id=LMN456&export=download", "LMN456"),
        ("https://drive.google.com/uc?export=view&id=QWE987", "QWE987"),
    ],
)
def test_drive_view_to_direct_variants(url: str, expected_id: str) -> None:
    direct = drive_view_to_direct(url)
    assert direct == f"https://drive.google.com/uc?export=view&id={expected_id}"


def test_drive_view_to_direct_trims_invisible_chars() -> None:
    url = "\u200b https://drive.google.com/file/d/ABC123/view "
    assert drive_view_to_direct(url) == "https://drive.google.com/uc?export=view&id=ABC123"


def test_drive_view_to_direct_rejects_folder() -> None:
    folder_url = "https://drive.google.com/drive/folders/1AbCdEf"
    with pytest.raises(DriveFolderUrlError):
        drive_view_to_direct(folder_url)


def test_drive_view_to_direct_invalid_url() -> None:
    with pytest.raises(DriveUrlError):
        drive_view_to_direct("https://example.com/not-drive")


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
            batch = await catalog.pick_batch(
                gender="female",
                batch_size=2,
                scheme="GENDER_OR_GENDER_UNISEX",
                seen_ids=seen_first,
                rng=random.Random(42),
            )
            assert batch.items
            assert batch.items[0].unique_id != female_models[0].unique_id
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


def test_catalog_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    csv_text = (
        "Название,Ссылка на сайт,Ссылка на изображение для пользователя\n"
        "Valid,https://example.com/a,https://drive.google.com/file/d/AAA/view\n"
        "Empty,https://example.com/b,   \n"
        "Folder,https://example.com/c,https://drive.google.com/drive/folders/FFF?usp=sharing\n"
        "Bad,https://example.com/d,https://example.com/not-drive\n"
    )

    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=csv_text)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url="https://example.com/logs",
                    cache_ttl_seconds=60,
                    retries=3,
                ),
                client=client,
            )
            with caplog.at_level(logging.INFO):
                await catalog.list_by_gender("male")
            await catalog.aclose()

    caplog.clear()
    asyncio.run(_run())
    summary_logs = [record for record in caplog.records if "Catalog CSV parsed" in record.message]
    assert summary_logs, "Expected summary log entry"
    message = summary_logs[-1].message
    assert "total_rows=4" in message
    assert "valid_rows=1" in message
    assert "skipped_empty=1" in message
    assert "skipped_folder=1" in message
    assert "skipped_invalid=1" in message
