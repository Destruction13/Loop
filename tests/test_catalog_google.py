"""Tests for Google Sheets catalog integration."""

from __future__ import annotations

import asyncio

import httpx

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
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = GoogleSheetCatalog(GoogleCatalogConfig(csv_url="https://example.com"), client=client)
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
