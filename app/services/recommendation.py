"""Recommendation service implementing gender/unisex batching."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum

from app.models import GlassModel
from app.services.catalog_base import CatalogService
from app.services.repository import Repository


class PickScheme(Enum):
    """Available schemes for selecting recommendation batches."""

    GENDER_OR_GENDER_UNISEX = "GENDER_OR_GENDER_UNISEX"

    @classmethod
    def from_string(cls, value: str | None) -> "PickScheme":
        normalized = (value or cls.GENDER_OR_GENDER_UNISEX.value).strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown pick scheme: {value}")


@dataclass(slots=True)
class RecommendationSettings:
    """Configuration options controlling recommendation batching."""

    batch_size: int
    pick_scheme: PickScheme
    clear_on_catalog_change: bool


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
        self._scheme = self._settings.pick_scheme

    async def recommend_for_user(
        self, user_id: int, selected_gender: str
    ) -> RecommendationResult:
        """Return models for the user based on configured quotas."""

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
        normalized_gender = self._normalize_group_key(
            selected_gender, log_unknown=False
        )
        batch = await self._catalog.pick_batch(
            gender=selected_gender,
            batch_size=self._batch_size,
            scheme=self._scheme.value,
            rng=self._rng,
            snapshot=snapshot,
        )

        picks = list(batch.items)
        exhausted = batch.exhausted

        if not picks:
            return RecommendationResult(models=[], exhausted=exhausted)
        self._logger.debug(
            "Selected models for user %s: %s",
            user_id,
            [model.unique_id for model in picks],
        )
        return RecommendationResult(models=picks, exhausted=exhausted)

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


__all__ = [
    "RecommendationService",
    "RecommendationSettings",
    "RecommendationResult",
    "PickScheme",
]
