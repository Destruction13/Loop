from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import List

from app.models import GlassModel
from app.services.catalog_base import CatalogBatch, CatalogService, CatalogSnapshot
from app.services.recommendation import PickScheme, RecommendationService, RecommendationSettings
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

    async def list_by_gender(self, gender: str) -> List[GlassModel]:  # noqa: D401 - not used
        normalized = _normalize_gender(gender)
        if normalized == "Унисекс":
            allowed = {"Унисекс"}
        else:
            allowed = {normalized, "Унисекс"}
        return [model for model in self._snapshot.models if model.gender in allowed]

    async def pick_batch(
        self,
        *,
        gender: str,
        batch_size: int,
        scheme: str,
        rng: random.Random | None = None,
        snapshot: CatalogSnapshot | None = None,
    ) -> CatalogBatch:
        rng = rng or random.Random()
        snapshot = snapshot or self._snapshot
        unique_models: dict[str, GlassModel] = {}
        for model in snapshot.models:
            unique_models.setdefault(model.unique_id, model)
        models = list(unique_models.values())

        normalized_gender = _normalize_gender(gender)
        gender_pool = [
            model
            for model in models
            if _normalize_gender(model.gender) == normalized_gender
        ]
        unisex_pool = [
            model
            for model in models
            if _normalize_gender(model.gender) == "Унисекс"
        ]

        if normalized_gender == "Унисекс":
            picks, exhausted = _pick_unisex_batch(rng, unisex_pool, batch_size)
        else:
            picks, exhausted = _pick_gender_batch(
                rng, gender_pool, unisex_pool, batch_size, scheme
            )
        return CatalogBatch(items=picks, exhausted=exhausted)


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


def _normalize_gender(value: str) -> str:
    prepared = (value or "").strip().lower()
    if prepared.startswith("муж") or prepared in {"m", "male"}:
        return "Мужской"
    if prepared.startswith("жен") or prepared in {"f", "female"}:
        return "Женский"
    if prepared.startswith("уни") or prepared.startswith("uni") or prepared in {"u", "unisex"}:
        return "Унисекс"
    return "Other"


def _sample(rng: random.Random, items: Iterable[GlassModel], count: int) -> list[GlassModel]:
    pool = list(items)
    if count <= 0:
        return []
    if len(pool) <= count:
        return list(pool)
    return rng.sample(pool, count)


def _pick_unisex_batch(
    rng: random.Random, pool: list[GlassModel], batch_size: int
) -> tuple[list[GlassModel], bool]:
    selection = _sample(rng, pool, min(batch_size, len(pool)))
    used = {model.unique_id for model in selection}
    remaining = [model for model in pool if model.unique_id not in used]
    return selection, len(remaining) < batch_size


def _pick_gender_batch(
    rng: random.Random,
    gender_pool: list[GlassModel],
    unisex_pool: list[GlassModel],
    batch_size: int,
    scheme: str,
) -> tuple[list[GlassModel], bool]:
    normalized_scheme = (scheme or "GENDER_OR_GENDER_UNISEX").strip().upper()
    picks: list[GlassModel] = []
    used: set[str] = set()

    schemes: list[str] = []
    if len(gender_pool) >= 2:
        schemes.append("GG")
    if len(gender_pool) >= 1 and len(unisex_pool) >= 1:
        schemes.append("GU")

    chosen: str | None = None
    if normalized_scheme == "GENDER_OR_GENDER_UNISEX" and schemes:
        chosen = schemes[0] if len(schemes) == 1 else rng.choice(schemes)

    if chosen == "GG":
        gender_selection = _sample(rng, gender_pool, min(2, len(gender_pool)))
        picks.extend(gender_selection)
        used.update(model.unique_id for model in gender_selection)
    elif chosen == "GU":
        gender_selection = _sample(rng, gender_pool, 1)
        picks.extend(gender_selection)
        used.update(model.unique_id for model in gender_selection)
        remaining_unisex = [model for model in unisex_pool if model.unique_id not in used]
        unisex_selection = _sample(rng, remaining_unisex, 1)
        picks.extend(unisex_selection)
        used.update(model.unique_id for model in unisex_selection)
    else:
        if gender_pool:
            gender_selection = _sample(rng, gender_pool, min(batch_size, len(gender_pool)))
            picks.extend(gender_selection)
            used.update(model.unique_id for model in gender_selection)
        elif unisex_pool:
            unisex_selection = _sample(rng, unisex_pool, min(batch_size, len(unisex_pool)))
            picks.extend(unisex_selection)
            used.update(model.unique_id for model in unisex_selection)

    if len(picks) < batch_size and unisex_pool:
        remaining = [model for model in unisex_pool if model.unique_id not in used]
        if remaining:
            extra = _sample(rng, remaining, min(batch_size - len(picks), len(remaining)))
            picks.extend(extra)
            used.update(model.unique_id for model in extra)

    if len(picks) < batch_size and gender_pool:
        remaining = [model for model in gender_pool if model.unique_id not in used]
        if remaining:
            extra = _sample(rng, remaining, min(batch_size - len(picks), len(remaining)))
            picks.extend(extra)
            used.update(model.unique_id for model in extra)

    remaining_gender = [model for model in gender_pool if model.unique_id not in used]
    remaining_unisex = [model for model in unisex_pool if model.unique_id not in used]
    exhausted = (len(remaining_gender) + len(remaining_unisex)) < batch_size
    return picks, exhausted


def test_pick_scheme_gender_or_unisex(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "scheme.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "Мужской"),
            make_model("m2", "Мужской"),
            make_model("m3", "Мужской"),
            make_model("u1", "Унисекс"),
            make_model("u2", "Унисекс"),
        ]
        catalog = DummyCatalog(models)

        settings = RecommendationSettings(
            batch_size=2,
            pick_scheme=PickScheme.GENDER_OR_GENDER_UNISEX,
            clear_on_catalog_change=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(7),
        )

        counts = {"GG": 0, "GU": 0}
        for _ in range(200):
            batch = await catalog.pick_batch(
                gender="male",
                batch_size=2,
                scheme="GENDER_OR_GENDER_UNISEX",
                rng=random.Random(_),
                snapshot=await catalog.snapshot(),
            )
            genders = {model.gender for model in batch.items}
            if genders == {"Мужской"}:
                counts["GG"] += 1
            elif "Унисекс" in genders:
                counts["GU"] += 1
        assert counts["GG"] > 0 and counts["GU"] > 0
        assert abs(counts["GG"] - counts["GU"]) < 60

        # Fallback when no unisex available
        catalog.update([make_model("m4", "Мужской"), make_model("m5", "Мужской")], "v2")
        batch = await catalog.pick_batch(
            gender="male",
            batch_size=2,
            scheme="GENDER_OR_GENDER_UNISEX",
            rng=random.Random(1),
        )
        assert all(model.gender == "Мужской" for model in batch.items)

        # Fallback when only unisex remains
        catalog.update([make_model("u9", "Унисекс")], "v3")
        batch = await catalog.pick_batch(
            gender="male",
            batch_size=2,
            scheme="GENDER_OR_GENDER_UNISEX",
            rng=random.Random(2),
        )
        assert len(batch.items) == 1
        assert batch.exhausted is True
        assert batch.items[0].gender == "Унисекс"

    asyncio.run(scenario())


def test_recommendation_allows_repeats(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "repeats.db", daily_limit=5)
        await repo.init()
        models = [make_model("m1", "Мужской")]
        catalog = DummyCatalog(models)
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=RecommendationSettings(
                batch_size=2,
                pick_scheme=PickScheme.GENDER_OR_GENDER_UNISEX,
                clear_on_catalog_change=False,
            ),
            rng=random.Random(0),
        )

        first = await service.recommend_for_user(10, "male")
        second = await service.recommend_for_user(10, "male")

        assert first.models and second.models
        assert first.models[0].unique_id == "m1"
        assert second.models[0].unique_id == "m1"

    asyncio.run(scenario())


def test_catalog_version_change_clears_history(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "version.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "Мужской"),
            make_model("m2", "Мужской"),
            make_model("u1", "Унисекс"),
        ]
        catalog = DummyCatalog(models, version_hash="v1")
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=RecommendationSettings(
                batch_size=2,
                pick_scheme=PickScheme.GENDER_OR_GENDER_UNISEX,
                clear_on_catalog_change=True,
            ),
            rng=random.Random(5),
        )

        first = await service.recommend_for_user(99, "male")
        assert first.models
        catalog.update(models, version_hash="v2")
        second = await service.recommend_for_user(99, "male")
        assert second.models

    asyncio.run(scenario())


def test_recommendation_respects_unisex_selection(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "unisex.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("u1", "Унисекс"),
            make_model("u2", "Унисекс"),
            make_model("u3", "Унисекс"),
        ]
        catalog = DummyCatalog(models)
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=RecommendationSettings(
                batch_size=2,
                pick_scheme=PickScheme.GENDER_OR_GENDER_UNISEX,
                clear_on_catalog_change=False,
            ),
            rng=random.Random(11),
        )

        result = await service.recommend_for_user(44, "unisex")
        assert len(result.models) == 2
        assert all(model.gender == "Унисекс" for model in result.models)

    asyncio.run(scenario())

