"""Catalog service interface."""

from __future__ import annotations

import abc
from typing import Iterable, List

from app.models import FilterOptions, ModelItem


class CatalogService(abc.ABC):
    """Interface for catalog operations."""

    @abc.abstractmethod
    async def list_models(self, filters: FilterOptions) -> List[ModelItem]:
        """Return models matching filters."""

    @abc.abstractmethod
    async def pick_four(self, user_id: int, filters: FilterOptions) -> List[ModelItem]:
        """Return up to four models taking into account seen history."""

    @abc.abstractmethod
    async def mark_seen(self, user_id: int, model_ids: Iterable[str]) -> None:
        """Store that user has seen specific models."""
