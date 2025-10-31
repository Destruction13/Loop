from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Iterable, List

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
        if normalized == "unisex":
            allowed = {"unisex"}
        elif normalized in {"male", "female"}:
            allowed = {normalized, "unisex"}
        else:
            allowed = {normalized, "unisex"}
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
        normalized_scheme = (scheme or "GENDER_AND_UNISEX_ONLY").strip().upper()
        gender_pool = [
            model
            for model in models
            if _normalize_gender(model.gender) == normalized_gender
        ]
        unisex_pool = [
            model
            for model in models
            if _normalize_gender(model.gender) == "unisex"
        ]

        if normalized_scheme == "UNIVERSAL":
            picks, exhausted = _pick_universal_batch(
                rng,
                gender_pool,
                unisex_pool,
                batch_size,
                normalized_gender,
            )
        elif normalized_gender == "unisex":
            picks, exhausted = _pick_unisex_batch(rng, unisex_pool, batch_size)
        else:
            picks, exhausted = _pick_gender_batch(
                rng, gender_pool, unisex_pool, batch_size, normalized_scheme
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
        return "male"
    if prepared.startswith("жен") or prepared in {"f", "female"}:
        return "female"
    if prepared.startswith("уни") or prepared.startswith("uni") or prepared in {"u", "unisex"}:
        return "unisex"
    return "other"


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
    return selection, len(selection) == 0


def _pick_gender_batch(
    rng: random.Random,
    gender_pool: list[GlassModel],
    unisex_pool: list[GlassModel],
    batch_size: int,
    scheme: str,
) -> tuple[list[GlassModel], bool]:
    normalized_scheme = (scheme or "GENDER_AND_UNISEX_ONLY").strip().upper()
    picks: list[GlassModel] = []
    used: set[str] = set()

    schemes: list[str] = []
    if len(gender_pool) >= 2:
        schemes.append("GG")
    if len(gender_pool) >= 1 and len(unisex_pool) >= 1:
        schemes.append("GU")

    chosen: str | None = None
    if normalized_scheme == "GENDER_AND_GENDER_ONLY":
        if "GG" in schemes:
            chosen = "GG"
    elif normalized_scheme == "GENDER_AND_UNISEX_ONLY":
        if "GU" in schemes:
            chosen = "GU"
    elif normalized_scheme == "GENDER_OR_GENDER_UNISEX" and schemes:
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

    exhausted = len(picks) == 0
    return picks, exhausted


def _pick_universal_batch(
    rng: random.Random,
    gender_pool: list[GlassModel],
    unisex_pool: list[GlassModel],
    batch_size: int,
    normalized_gender: str,
) -> tuple[list[GlassModel], bool]:
    if normalized_gender == "unisex":
        allowed = list(unisex_pool)
    else:
        allowed = list(gender_pool)
        allowed.extend(unisex_pool)

    selection = _sample(rng, allowed, min(batch_size, len(allowed)))
    exhausted = len(selection) == 0
    return selection, exhausted


def test_pick_scheme_gender_or_unisex(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "scheme.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "male"),
            make_model("m2", "male"),
            make_model("m3", "male"),
            make_model("u1", "unisex"),
            make_model("u2", "unisex"),
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
            if genders == {"male"}:
                counts["GG"] += 1
            elif "unisex" in genders:
                counts["GU"] += 1
        assert counts["GG"] > 0 and counts["GU"] > 0
        assert abs(counts["GG"] - counts["GU"]) < 60

        # Fallback when no unisex available
        catalog.update([make_model("m4", "male"), make_model("m5", "male")], "v2")
        batch = await catalog.pick_batch(
            gender="male",
            batch_size=2,
            scheme="GENDER_OR_GENDER_UNISEX",
            rng=random.Random(1),
        )
        assert all(model.gender == "male" for model in batch.items)

        # Fallback when only unisex remains
        catalog.update([make_model("u9", "unisex")], "v3")
        batch = await catalog.pick_batch(
            gender="male",
            batch_size=2,
            scheme="GENDER_OR_GENDER_UNISEX",
            rng=random.Random(2),
        )
        assert len(batch.items) == 1
        assert batch.exhausted is False
        assert batch.items[0].gender == "unisex"

    asyncio.run(scenario())


def test_pick_scheme_gender_and_unisex_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "scheme_gu.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "male"),
            make_model("m2", "male"),
            make_model("u1", "unisex"),
        ]
        catalog = DummyCatalog(models)

        settings = RecommendationSettings(
            batch_size=2,
            pick_scheme=PickScheme.GENDER_AND_UNISEX_ONLY,
            clear_on_catalog_change=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(3),
        )

        result = await service.recommend_for_user(1, "male")
        genders = {model.gender for model in result.models}
        assert genders == {"male", "unisex"}

        catalog.update([make_model("m3", "male")], "v2")
        fallback = await service.recommend_for_user(1, "male")
        assert all(model.gender == "male" for model in fallback.models)

    asyncio.run(scenario())


def test_pick_scheme_gender_and_gender_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "scheme_gg.db", daily_limit=5)
        await repo.init()
        models = [
            make_model("m1", "male"),
            make_model("m2", "male"),
            make_model("u1", "unisex"),
        ]
        catalog = DummyCatalog(models)

        settings = RecommendationSettings(
            batch_size=2,
            pick_scheme=PickScheme.GENDER_AND_GENDER_ONLY,
            clear_on_catalog_change=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(4),
        )

        result = await service.recommend_for_user(2, "male")
        assert all(model.gender == "male" for model in result.models)

        catalog.update([make_model("u2", "unisex")], "v2")
        fallback = await service.recommend_for_user(2, "male")
        assert fallback.models
        assert all(model.gender == "unisex" for model in fallback.models)

    asyncio.run(scenario())


def test_pick_scheme_universal(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "scheme_universal.db", daily_limit=5)
        await repo.init()
        models = [
            *[make_model(f"m{i}", "male") for i in range(1, 5)],
            *[make_model(f"f{i}", "female") for i in range(1, 4)],
            *[make_model(f"u{i}", "unisex") for i in range(1, 7)],
        ]
        catalog = DummyCatalog(models)

        settings = RecommendationSettings(
            batch_size=2,
            pick_scheme=PickScheme.UNIVERSAL,
            clear_on_catalog_change=False,
        )
        service = RecommendationService(
            catalog=catalog,
            repository=repo,
            settings=settings,
            rng=random.Random(6),
        )

        for seed in range(5):
            batch = await catalog.pick_batch(
                gender="male",
                batch_size=2,
                scheme="UNIVERSAL",
                rng=random.Random(seed),
                snapshot=await catalog.snapshot(),
            )
            assert len(batch.items) == 2
            assert all(model.gender in {"male", "unisex"} for model in batch.items)

        female_batch = await catalog.pick_batch(
            gender="female",
            batch_size=2,
            scheme="UNIVERSAL",
            rng=random.Random(11),
            snapshot=await catalog.snapshot(),
        )
        assert len(female_batch.items) == 2
        assert all(model.gender in {"female", "unisex"} for model in female_batch.items)
        assert all(model.gender != "male" for model in female_batch.items)

        unisex_batch = await catalog.pick_batch(
            gender="unisex",
            batch_size=2,
            scheme="UNIVERSAL",
            rng=random.Random(12),
            snapshot=await catalog.snapshot(),
        )
        assert len(unisex_batch.items) == 2
        assert all(model.gender == "unisex" for model in unisex_batch.items)

        male_result = await service.recommend_for_user(10, "male")
        assert male_result.models
        assert all(model.gender in {"male", "unisex"} for model in male_result.models)

        female_result = await service.recommend_for_user(11, "female")
        assert female_result.models
        assert all(model.gender in {"female", "unisex"} for model in female_result.models)
        assert all(model.gender != "male" for model in female_result.models)

        unisex_result = await service.recommend_for_user(12, "unisex")
        assert unisex_result.models
        assert all(model.gender == "unisex" for model in unisex_result.models)

    asyncio.run(scenario())


def test_recommendation_allows_repeats(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = Repository(tmp_path / "repeats.db", daily_limit=5)
        await repo.init()
        models = [make_model("m1", "male")]
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
            make_model("m1", "male"),
            make_model("m2", "male"),
            make_model("u1", "unisex"),
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
            make_model("u1", "unisex"),
            make_model("u2", "unisex"),
            make_model("u3", "unisex"),
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
        assert all(model.gender == "unisex" for model in result.models)

    asyncio.run(scenario())

