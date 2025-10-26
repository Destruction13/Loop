"""Catalog service interface."""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass
from typing import List

from app.models import GlassModel


@dataclass(slots=True)
class CatalogSnapshot:
    """Immutable view of the catalog data."""

    models: List[GlassModel]
    version_hash: str


@dataclass(slots=True)
class CatalogBatch:
    """Result of a catalog pick operation."""

    items: List[GlassModel]
    exhausted: bool


class CatalogError(RuntimeError):
    """Raised when catalog data cannot be retrieved."""


class CatalogService(abc.ABC):
    """Interface for catalog operations."""

    @abc.abstractmethod
    async def list_by_gender(self, gender: str) -> list[GlassModel]:
        """Return catalog entries filtered by gender."""

    async def aclose(self) -> None:
        """Optional hook for graceful shutdown."""

        return None

    @abc.abstractmethod
    async def snapshot(self) -> CatalogSnapshot:
        """Return cached catalog data with version hash."""

    async def pick_batch(
        self,
        *,
        gender: str,
        batch_size: int,
        scheme: str,
        rng: random.Random | None = None,
        snapshot: CatalogSnapshot | None = None,
    ) -> CatalogBatch:
        """Pick a batch of models using the provided scheme."""

        raise NotImplementedError
