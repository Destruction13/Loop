"""Local filesystem storage implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.storage_base import StorageService
from app.utils.paths import ensure_dir, sanitize_filename


class LocalStorage(StorageService):
    """Store files locally on disk."""

    def __init__(self, uploads_root: Path, results_root: Path) -> None:
        self._uploads_root = uploads_root
        self._results_root = results_root

    async def allocate_upload_path(self, user_id: int, filename: str) -> Path:
        safe = sanitize_filename(filename)
        path = ensure_dir(self._uploads_root / str(user_id)) / safe
        await asyncio.to_thread(lambda: None)
        return path

    async def allocate_result_dir(self, user_id: int, session_id: str) -> Path:
        path = ensure_dir(self._results_root / str(user_id) / session_id)
        await asyncio.to_thread(lambda: None)
        return path
