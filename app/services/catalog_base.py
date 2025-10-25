"""Catalog service interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, List

from app.models import GlassModel


@dataclass(slots=True)
class CatalogSnapshot:
    """Immutable view of the catalog data."""

    models: List[GlassModel]
    version_hash: str


class CatalogError(RuntimeError):
    """Raised when catalog data cannot be retrieved."""


class CatalogService(abc.ABC):
    """Interface for catalog operations."""

    @abc.abstractmethod
    async def list_by_gender(self, gender: str) -> list[GlassModel]:
        """Return catalog entries filtered by gender."""

    @abc.abstractmethod
    async def pick_four(self, gender: str, seen_ids: Iterable[str]) -> list[GlassModel]:
        """Return up to four models avoiding already seen identifiers when possible."""

    async def aclose(self) -> None:
        """Optional hook for graceful shutdown."""

        return None

    @abc.abstractmethod
    async def snapshot(self) -> CatalogSnapshot:
        """Return cached catalog data with version hash."""
