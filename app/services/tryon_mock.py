"""Mock try-on service generating white images."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, ImageDraw

from app.services.storage_base import StorageService
from app.services.tryon_base import TryOnService


class MockTryOnService(TryOnService):
    """Generate placeholder try-on results."""

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def generate(
        self,
        user_id: int,
        session_id: str,
        input_photo: Path,
        overlays: Iterable[Optional[Path]],
        count: int = 4,
    ) -> List[Path]:
        output_dir = await self._storage.allocate_result_dir(user_id, session_id)
        return await asyncio.to_thread(
            self._generate_sync, output_dir, input_photo, count
        )

    def _generate_sync(self, output_dir: Path, input_photo: Path, count: int) -> List[Path]:
        if input_photo.exists():
            with Image.open(input_photo) as img:
                size = img.size
        else:
            size = (1024, 1024)
        results: List[Path] = []
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        for idx in range(1, count + 1):
            image = Image.new("RGB", size, color="white")
            draw = ImageDraw.Draw(image)
            text = f"DEMO {timestamp}" 
            draw.text((10, 10), text, fill="black")
            output_path = output_dir / f"result-{idx}.jpg"
            image.save(output_path, format="JPEG", quality=85)
            results.append(output_path)
        return results
