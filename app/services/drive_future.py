"""Placeholder for future Google Drive integration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DrivePublisher(Protocol):
    """Protocol for uploading files to Google Drive."""

    async def publish(self, local_path: Path) -> str:
        """Upload file and return public URL."""
        ...


class GoogleDriveService:
    """Stub implementation raising NotImplementedError."""

    async def publish(self, local_path: Path) -> str:
        raise NotImplementedError("Google Drive integration is not implemented yet")
