"""Local filesystem catalog implementation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from app.models import FilterOptions, ModelItem, ModelMeta
from app.services.catalog_base import CatalogService
from app.services.repository import Repository
from app.utils.pick import pick_four
from app.utils.paths import ensure_dir

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - hard failure in runtime environment
    raise RuntimeError("Pillow is required for catalog placeholder generation") from exc


@dataclass
class CatalogLocalConfig:
    """Configuration for local catalog service."""

    root: Path


class CatalogServiceLocal(CatalogService):
    """Read catalog data from local filesystem."""

    def __init__(self, config: CatalogLocalConfig, repository: Repository) -> None:
        self._config = config
        self._repository = repository

    async def list_models(self, filters: FilterOptions) -> List[ModelItem]:
        return await asyncio.to_thread(self._load_models, filters)

    async def pick_four(self, user_id: int, filters: FilterOptions) -> List[ModelItem]:
        models = await self.list_models(filters)
        profile = await self._repository.get_user(user_id)
        seen = profile.seen_models if profile else []
        picked = pick_four(models, seen)
        await self.mark_seen(user_id, [model.model_id for model in picked])
        return picked

    async def mark_seen(self, user_id: int, model_ids: Iterable[str]) -> None:
        await self._repository.add_seen_models(user_id, list(model_ids))

    # internal helpers
    def _load_models(self, filters: FilterOptions) -> List[ModelItem]:
        base_path = self._config.root / filters.gender / filters.age_bucket
        candidates: List[ModelItem] = []
        styles: List[Path]
        if filters.style and (base_path / filters.style).exists():
            styles = [base_path / filters.style]
        else:
            styles = [path for path in base_path.iterdir() if path.is_dir()] if base_path.exists() else []
        for style_path in styles:
            for model_dir in style_path.iterdir():
                if not model_dir.is_dir():
                    continue
                model_id = model_dir.name
                meta = self._read_meta(model_dir / "meta.json")
                thumb = self._resolve_thumb(model_dir)
                overlay = model_dir / "overlay.png"
                if not overlay.exists():
                    overlay = None
                candidates.append(
                    ModelItem(
                        model_id=model_id,
                        thumb_path=thumb,
                        overlay_path=overlay,
                        meta=meta,
                    )
                )
        return candidates

    def _resolve_thumb(self, model_dir: Path) -> Path:
        for name in ("thumb.jpg", "thumb.png"):
            thumb = model_dir / name
            if thumb.exists():
                return thumb
        auto_thumb = model_dir / "thumb_auto.jpg"
        if not auto_thumb.exists():
            self._create_placeholder(auto_thumb)
        return auto_thumb

    def _create_placeholder(self, destination: Path) -> None:
        ensure_dir(destination.parent)
        image = Image.new("RGB", (512, 512), color="#f4f4f4")
        draw = ImageDraw.Draw(image)
        text = "Нет превью"
        text_color = "#666666"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        width = text_bbox[2] - text_bbox[0]
        height = text_bbox[3] - text_bbox[1]
        position = ((image.width - width) / 2, (image.height - height) / 2)
        draw.text(position, text, fill=text_color, font=font)
        image.save(destination, format="JPEG")

    def _read_meta(self, meta_path: Path) -> ModelMeta:
        if not meta_path.exists():
            return ModelMeta(title="Без названия", product_url="https://example.com")
        with meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return ModelMeta(
            title=data.get("title", "Без названия"),
            product_url=data.get("product_url", "https://example.com"),
            shape=data.get("shape"),
            brand=data.get("brand"),
        )
