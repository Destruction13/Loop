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

    batch_size: int
    pick_rule: str
    unique_scope: UniqueScope
    clear_on_catalog_change: bool
    topup_from_any: bool


@dataclass(slots=True)
class RecommendationResult:
    """Result of a recommendation lookup."""

    models: list[GlassModel]
    exhausted: bool


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
        self._batch_size = max(self._settings.batch_size, 1)
        self._gender_quota, self._unisex_quota = self._parse_pick_rule(
            self._settings.pick_rule
        )

    async def recommend_for_user(
        self, user_id: int, selected_gender: str
    ) -> RecommendationResult:
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
            group = self._normalize_group_key(
                model.gender, log_unknown=True, model_id=model.unique_id
            )
            if group == selected_key:
                available_gender.append(model)
            elif group == "унисекс":
                available_unisex.append(model)
            else:
                available_other.append(model)

        available_gender = [m for m in available_gender if m.unique_id not in seen_ids]
        available_unisex = [m for m in available_unisex if m.unique_id not in seen_ids]
        available_other = [m for m in available_other if m.unique_id not in seen_ids]

        if selected_key != "унисекс" and len(available_gender) < self._gender_quota:
            self._logger.info(
                "not enough candidates for gender (need %s, got %s)",
                self._gender_quota,
                len(available_gender),
            )
        if len(available_unisex) < self._unisex_quota:
            self._logger.info(
                "not enough candidates for unisex (need %s, got %s)",
                self._unisex_quota,
                len(available_unisex),
            )

        picks: list[GlassModel] = []
        picked_ids: set[str] = set()

        if selected_key == "унисекс":
            unisex_selection = self._sample(
                available_unisex,
                min(self._batch_size, len(available_unisex)),
            )
            picks.extend(unisex_selection)
            picked_ids.update(model.unique_id for model in unisex_selection)
        else:
            gender_target = min(self._gender_quota, self._batch_size)
            gender_selection = self._sample(
                available_gender,
                min(gender_target, len(available_gender)),
            )
            picks.extend(gender_selection)
            picked_ids.update(model.unique_id for model in gender_selection)

            remaining_slots = max(self._batch_size - len(picks), 0)
            unisex_target = min(self._unisex_quota, remaining_slots)
            unisex_candidates = [
                model for model in available_unisex if model.unique_id not in picked_ids
            ]
            unisex_selection = self._sample(
                unisex_candidates,
                min(unisex_target, len(unisex_candidates)),
            )
            picks.extend(unisex_selection)
            picked_ids.update(model.unique_id for model in unisex_selection)

            remaining_slots = max(self._batch_size - len(picks), 0)
            if remaining_slots > 0:
                fill_pool: list[GlassModel] = []
                fill_pool.extend(
                    model
                    for model in available_gender
                    if model.unique_id not in picked_ids
                )
                fill_pool.extend(
                    model
                    for model in available_unisex
                    if model.unique_id not in picked_ids
                )
                if self._settings.topup_from_any:
                    fill_pool.extend(
                        model
                        for model in available_other
                        if model.unique_id not in picked_ids
                    )
                if fill_pool:
                    topup_selection = self._sample(
                        fill_pool, min(remaining_slots, len(fill_pool))
                    )
                    picks.extend(topup_selection)
                    picked_ids.update(model.unique_id for model in topup_selection)

        exhausted = False
        if selected_key == "унисекс":
            exhausted = len(available_unisex) < self._batch_size
        else:
            exhausted = (len(available_gender) + len(available_unisex)) < self._batch_size

        if not picks:
            return RecommendationResult(models=[], exhausted=exhausted)

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
        return RecommendationResult(models=shuffled, exhausted=exhausted)

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

    def _parse_pick_rule(self, rule: str) -> tuple[int, int]:
        raw = (rule or "").strip()
        if not raw:
            return self._batch_size, 0
        parts = raw.split("_", 1)
        try:
            primary = int(parts[0])
        except ValueError:
            primary = self._batch_size
        if len(parts) > 1:
            try:
                secondary = int(parts[1])
            except ValueError:
                secondary = 0
        else:
            secondary = 0
        return max(primary, 0), max(secondary, 0)


__all__ = [
    "RecommendationService",
    "RecommendationSettings",
    "RecommendationResult",
    "UniqueScope",
]
