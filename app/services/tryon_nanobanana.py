"""Placeholder for NanoBanana API integration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from app.services.tryon_base import TryOnService


class NanoBananaTryOnService(TryOnService):
    """Stub for real NanoBanana API calls."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self._api_url = api_url
        self._api_key = api_key

    async def generate(
        self,
        user_id: int,
        session_id: str,
        input_photo: Path,
        overlays: Iterable[Optional[Path]],
        count: int = 4,
    ) -> List[Path]:
        raise NotImplementedError("NanoBanana API integration is not implemented yet")
