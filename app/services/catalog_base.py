"""Catalog service interface."""

from __future__ import annotations

import abc
from typing import Iterable

from app.models import GlassModel


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
