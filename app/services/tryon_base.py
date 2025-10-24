"""Try-on service interface."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterable, List, Optional


class TryOnService(abc.ABC):
    """Interface for generating try-on results."""

    @abc.abstractmethod
    async def generate(
        self,
        user_id: int,
        session_id: str,
        input_photo: Path,
        overlays: Iterable[Optional[Path]],
        count: int = 4,
    ) -> List[Path]:
        """Generate try-on images."""
