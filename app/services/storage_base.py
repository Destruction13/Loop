"""Storage service interface."""

from __future__ import annotations

import abc
from pathlib import Path


class StorageService(abc.ABC):
    """Interface for managing file storage."""

    @abc.abstractmethod
    async def allocate_upload_path(self, user_id: int, filename: str) -> Path:
        """Return path where uploaded file should be stored."""

    @abc.abstractmethod
    async def allocate_result_dir(self, user_id: int, session_id: str) -> Path:
        """Return directory for storing generation results."""
