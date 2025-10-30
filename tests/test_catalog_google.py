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
        "Наименование,Модель,Ссылка,Sex,User Image,NanoBanana,Уникальный ID,Фото\n"
        "Alpha,A1,https://example.com/a,Мужской,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha,https://example.com/a.jpg\n"
        "Bravo,B1,https://example.com/b,Female,https://drive.google.com/file/d/BBB/view,https://nano/b,id-bravo,https://example.com/b.jpg\n"
        "Charlie,C1,https://example.com/c,U,https://drive.google.com/file/d/CCC/view,https://nano/c,id-charlie,https://example.com/c.jpg\n"
        ",,,male,https://drive.google.com/file/d/DDD/view,https://nano/d,id-missing,\n"
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
            assert male_models[1].gender == "unisex"
            female_models = await catalog.list_by_gender("female")
            assert [model.title for model in female_models] == ["Bravo", "Charlie"]
            batch = await catalog.pick_batch(
                gender="female",
                batch_size=2,
                scheme="GENDER_OR_GENDER_UNISEX",
                rng=random.Random(42),
            )
            assert batch.items
            await catalog.aclose()

    asyncio.run(_run())


def test_retry_policy_handles_redirect_and_server_errors() -> None:
    csv_text = (
        "Title,Model,Site,Gender,User Image,NanoBanana,UID\n"
        "Alpha,A1,https://example.com/a,male,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha\n"
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
        "Title,Model,Site,Gender,User Image,NanoBanana,UID\n"
        "Alpha,A1,https://example.com/a,male,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha\n"
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
        "Title,Model,Site,Gender,User Image,NanoBanana,UID\n"
        "Valid,V1,https://example.com/a,male,https://drive.google.com/file/d/AAA/view,https://nano/a,valid-1\n"
        ",,,,,,\n"
        "Folder,F1,https://example.com/c,female,https://drive.google.com/drive/folders/FFF?usp=sharing,https://nano/c,folder-1\n"
        "Bad,B1,ftp://example.com/d,other,https://example.com/not-drive,https://nano/d,bad-1\n"
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


def test_catalog_respects_row_limit() -> None:
    csv_text = (
        "Title,Model,Site,Gender,User Image,NanoBanana,UID\n"
        "Alpha,A1,https://example.com/a,male,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha\n"
        "Bravo,B1,https://example.com/b,male,https://drive.google.com/file/d/BBB/view,https://nano/b,id-bravo\n"
        "Charlie,C1,https://example.com/c,male,https://drive.google.com/file/d/CCC/view,https://nano/c,id-charlie\n"
    )

    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=csv_text)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url="https://example.com/limit",
                    cache_ttl_seconds=60,
                    retries=3,
                    parse_row_limit=1,
                ),
                client=client,
            )
            models = await catalog.list_by_gender("male")
            assert [model.title for model in models] == ["Alpha"]
            await catalog.aclose()

    asyncio.run(_run())


def test_direct_csv_url_used_without_fallback() -> None:
    csv_text = (
        "Title,Model,Site,Gender,User Image,NanoBanana,UID\n"
        "Alpha,A1,https://example.com/a,male,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha\n"
    )
    direct_url = (
        "https://docs.google.com/spreadsheets/d/e/ABC123/pub"
        "?gid=0&single=true&output=csv"
    )

    async def _run() -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            assert "output=csv" in str(request.url)
            return httpx.Response(200, text=csv_text)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url=direct_url,
                    cache_ttl_seconds=60,
                    retries=1,
                ),
                client=client,
            )
            models = await catalog.list_by_gender("male")
            assert models
            assert len(requested) == 1
            assert direct_url in requested[0]
            await catalog.aclose()

    asyncio.run(_run())


def test_sheet_id_fallback_respects_gid() -> None:
    csv_text = (
        "Title,Model,Site,Gender,User Image,NanoBanana,UID\n"
        "Alpha,A1,https://example.com/a,male,https://drive.google.com/file/d/AAA/view,https://nano/a,id-alpha\n"
    )

    requested: list[str] = []

    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            if request.url.path.endswith("/gviz/tq"):
                return httpx.Response(404, text="not found")
            return httpx.Response(200, text=csv_text)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            catalog = GoogleSheetCatalog(
                GoogleCatalogConfig(
                    csv_url=None,
                    sheet_id="SPREADSHEET123",
                    sheet_gid="42",
                    cache_ttl_seconds=60,
                    retries=1,
                ),
                client=client,
            )
            models = await catalog.list_by_gender("male")
            assert models
            await catalog.aclose()

        assert requested[0].startswith(
            "https://docs.google.com/spreadsheets/d/SPREADSHEET123/gviz/tq"
        )
        assert "gid=42" in requested[0]
        assert requested[1].startswith(
            "https://docs.google.com/spreadsheets/d/SPREADSHEET123/export"
        )
        assert "format=csv" in requested[1]
        assert "gid=42" in requested[1]

    asyncio.run(_run())
