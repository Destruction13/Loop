from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Iterable, List

from app.models import GlassModel
from app.services.catalog_base import CatalogService, CatalogSnapshot
from app.services.recommendation import (
    RecommendationService,
    RecommendationSettings,
    UniqueScope,
)
from app.services.repository import Repository


class DummyCatalog(CatalogService):
    def __init__(self, models: Iterable[GlassModel], version_hash: str = "v1") -> None:
        self._snapshot = CatalogSnapshot(models=list(models), version_hash=version_hash)

    def update(self, models: Iterable[GlassModel], version_hash: str) -> None:
        self._snapshot = CatalogSnapshot(models=list(models), version_hash=version_hash)

    async def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot(
            models=list(self._snapshot.models),
            version_hash=self._snapshot.version_hash,
        )

    async def list_by_gender(self, gender: str) -> List[GlassModel]:  # noqa: D401 - not used in tests
        return []

    async def pick_four(self, gender: str, seen_ids: Iterable[str]) -> List[GlassModel]:  # noqa: D401 - not used
        return []


def make_model(uid: str, gender: str) -> GlassModel:
    return GlassModel(
        unique_id=uid,
        title=f"Model {uid}",
        model_code=uid.upper(),
        site_url=f"https://example.com/{uid}",
        img_user_url=f"https://example.com/{uid}.jpg",
        img_nano_url="",
        gender=gender,
    )


def _count_by_gender(models: List[GlassModel], value: str) -> int:
    prefix = value.strip().lower()[:3]
    return sum(1 for model in models if (model.gender or "").strip().lower().startswith(prefix))


def test_recommendation_returns_expected_mix(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "rec1.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "Мужской"),
            make_model("m2", "Мужской"),
            make_model("m3", "Мужской"),
            make_model("u1", "Унисекс"),
            make_model("u2", "Унисекс"),
            make_model("u3", "Унисекс"),
            make_model("f1", "Женский"),
        ]
        catalog = DummyCatalog(models)
        settings = RecommendationSettings(
            batch_total=4,
            batch_gender=2,
            batch_unisex=2,
            unique_scope=UniqueScope.ALL,
            clear_on_catalog_change=False,
            topup_from_any=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(42),
        )

        picks = await service.recommend_for_user(1, "male")

        assert len(picks) == 4
        assert len({model.unique_id for model in picks}) == 4
        assert _count_by_gender(picks, "мужской") == 2
        assert _count_by_gender(picks, "унисекс") == 2

    asyncio.run(scenario())


def test_recommendation_enforces_uniqueness(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "rec2.db", daily_limit=5)
        await repo.init()
        models = [
            make_model(f"m{i}", "Мужской") for i in range(1, 5)
        ] + [
            make_model(f"u{i}", "Унисекс") for i in range(1, 5)
        ]
        catalog = DummyCatalog(models)
        settings = RecommendationSettings(
            batch_total=4,
            batch_gender=2,
            batch_unisex=2,
            unique_scope=UniqueScope.ALL,
            clear_on_catalog_change=False,
            topup_from_any=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(123),
        )

        first = await service.recommend_for_user(77, "male")
        second = await service.recommend_for_user(77, "male")

        assert not set(model.unique_id for model in first) & set(model.unique_id for model in second)

    asyncio.run(scenario())


def test_catalog_version_change_clears_history(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "rec3.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "Мужской"),
            make_model("m2", "Мужской"),
            make_model("u1", "Унисекс"),
            make_model("u2", "Унисекс"),
        ]
        catalog = DummyCatalog(models, version_hash="v1")
        settings = RecommendationSettings(
            batch_total=4,
            batch_gender=2,
            batch_unisex=2,
            unique_scope=UniqueScope.ALL,
            clear_on_catalog_change=True,
            topup_from_any=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(0),
        )

        first = await service.recommend_for_user(99, "male")
        assert first
        catalog.update(models, version_hash="v2")
        second = await service.recommend_for_user(99, "male")
        assert second

        # Without clearing the history the second batch would have been empty
        assert {model.unique_id for model in second}

    asyncio.run(scenario())


def test_recommendation_exhaustion_returns_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "rec4.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "Мужской"),
            make_model("m2", "Мужской"),
            make_model("u1", "Унисекс"),
            make_model("u2", "Унисекс"),
        ]
        catalog = DummyCatalog(models)
        settings = RecommendationSettings(
            batch_total=4,
            batch_gender=2,
            batch_unisex=2,
            unique_scope=UniqueScope.ALL,
            clear_on_catalog_change=False,
            topup_from_any=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(5),
        )

        first = await service.recommend_for_user(55, "male")
        assert first
        second = await service.recommend_for_user(55, "male")
        assert second == []

    asyncio.run(scenario())
