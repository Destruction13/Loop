"""Recommendation service implementing gender/unisex batching with uniqueness."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Sequence

from app.models import GlassModel
from app.services.catalog_base import CatalogService
from app.services.repository import Repository


class UniqueScope(Enum):
    """Window within which model recommendations must remain unique."""

    HOURS_24 = "24h"
    DAYS_7 = "7d"
    ALL = "all"

    @classmethod
    def from_string(cls, value: str) -> "UniqueScope":
        normalized = (value or "24h").strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown unique scope: {value}")

    def window_start(self, now: datetime) -> datetime | None:
        if self is UniqueScope.ALL:
            return None
        if self is UniqueScope.HOURS_24:
            return now - timedelta(hours=24)
        if self is UniqueScope.DAYS_7:
            return now - timedelta(days=7)
        raise ValueError(f"Unsupported scope: {self}")


@dataclass(slots=True)
class RecommendationSettings:
    """Configuration options controlling recommendation batching."""

    batch_total: int
    batch_gender: int
    batch_unisex: int
    unique_scope: UniqueScope
    clear_on_catalog_change: bool
    topup_from_any: bool


class RecommendationService:
    """Select catalog models for a user with gender/unisex quotas."""

    def __init__(
        self,
        catalog: CatalogService,
        repository: Repository,
        settings: RecommendationSettings,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._catalog = catalog
        self._repository = repository
        self._settings = settings
        self._rng = rng or random.Random()
        self._logger = logging.getLogger("loop_bot.recommendation")

    async def recommend_for_user(self, user_id: int, selected_gender: str) -> list[GlassModel]:
        """Return models for the user based on configured quotas and uniqueness."""

        now = datetime.now(timezone.utc)
        snapshot = await self._catalog.snapshot()
        changed, cleared = await self._repository.sync_catalog_version(
            snapshot.version_hash,
            clear_on_change=self._settings.clear_on_catalog_change,
        )
        if changed:
            action = "cleared" if cleared else "preserved"
            self._logger.info(
                "Catalog version updated to %s, history %s",
                snapshot.version_hash,
                action,
            )
        since = self._settings.unique_scope.window_start(now)
        seen_ids = await self._repository.list_seen_models(user_id, since=since)

        unique_models: dict[str, GlassModel] = {}
        for model in snapshot.models:
            unique_models.setdefault(model.unique_id, model)
        models = list(unique_models.values())

        selected_key = self._normalize_group_key(selected_gender, log_unknown=False)
        available_gender: list[GlassModel] = []
        available_unisex: list[GlassModel] = []
        available_other: list[GlassModel] = []

        for model in models:
            group = self._normalize_group_key(model.gender, log_unknown=True, model_id=model.unique_id)
            if group == selected_key:
                available_gender.append(model)
            elif group == "унисекс":
                available_unisex.append(model)
            else:
                available_other.append(model)

        available_gender = [m for m in available_gender if m.unique_id not in seen_ids]
        available_unisex = [m for m in available_unisex if m.unique_id not in seen_ids]
        available_other = [m for m in available_other if m.unique_id not in seen_ids]

        if len(available_gender) < self._settings.batch_gender:
            self._logger.info(
                "not enough candidates for gender (need %s, got %s)",
                self._settings.batch_gender,
                len(available_gender),
            )
        if len(available_unisex) < self._settings.batch_unisex:
            self._logger.info(
                "not enough candidates for unisex (need %s, got %s)",
                self._settings.batch_unisex,
                len(available_unisex),
            )

        picks: list[GlassModel] = []
        picked_ids: set[str] = set()
        target_total = max(self._settings.batch_total, 0)

        gender_quota = min(self._settings.batch_gender, target_total)
        gender_selection = self._sample(available_gender, gender_quota)
        picks.extend(gender_selection)
        picked_ids.update(model.unique_id for model in gender_selection)

        remaining_slots = max(target_total - len(picks), 0)
        unisex_quota = min(self._settings.batch_unisex, remaining_slots)
        unisex_candidates = [model for model in available_unisex if model.unique_id not in picked_ids]
        unisex_selection = self._sample(unisex_candidates, unisex_quota)
        picks.extend(unisex_selection)
        picked_ids.update(model.unique_id for model in unisex_selection)

        remaining_slots = max(target_total - len(picks), 0)
        if remaining_slots > 0 and self._settings.topup_from_any:
            pool: list[GlassModel] = []
            for source in (available_gender, available_unisex, available_other):
                for model in source:
                    if model.unique_id in picked_ids:
                        continue
                    pool.append(model)
            topup_selection = self._sample(pool, min(remaining_slots, len(pool)))
            picks.extend(topup_selection)
            picked_ids.update(model.unique_id for model in topup_selection)

        if not picks:
            return []

        shuffled = list(picks)
        self._rng.shuffle(shuffled)
        await self._repository.record_seen_models(
            user_id,
            [model.unique_id for model in shuffled],
            when=now,
        )
        self._logger.debug(
            "Selected models for user %s: %s",
            user_id,
            [model.unique_id for model in shuffled],
        )
        return shuffled

    def _normalize_group_key(
        self,
        value: str,
        *,
        log_unknown: bool,
        model_id: str | None = None,
    ) -> str:
        prepared = (value or "").strip().lower()
        male_tokens = {"муж", "мужской", "м", "male", "m"}
        female_tokens = {"жен", "женский", "ж", "female", "f"}
        unisex_tokens = {"унисекс", "uni", "unisex", "u"}
        if prepared in male_tokens or prepared.startswith("муж"):
            return "мужской"
        if prepared in female_tokens or prepared.startswith("жен"):
            return "женский"
        if prepared in unisex_tokens or prepared.startswith("уни") or prepared.startswith("uni"):
            return "унисекс"
        if prepared in {"other", "другое"}:
            return "other"
        if log_unknown and prepared:
            suffix = f" for model {model_id}" if model_id else ""
            self._logger.warning("Unknown gender value '%s'%s", value, suffix)
        return "other"

    def _sample(self, items: Sequence[GlassModel], count: int) -> list[GlassModel]:
        if count <= 0:
            return []
        if len(items) <= count:
            return list(items)
        return self._rng.sample(list(items), count)


__all__ = [
    "RecommendationService",
    "RecommendationSettings",
    "UniqueScope",
]
