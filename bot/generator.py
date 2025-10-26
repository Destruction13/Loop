from __future__ import annotations

import io
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneratorConfig:
    catalog_root: Path
    collage_width: int
    collage_height: int
    background_color: str
    margin: int
    divider_width: int
    divider_color: str


@dataclass(slots=True)
class ModelInfo:
    model_id: str
    title: str
    product_url: str
    gender: str
    metadata: dict[str, str]


@dataclass(slots=True)
class CollageResult:
    image: io.BytesIO
    models: list[ModelInfo]


class ModelCatalog:
    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config
        self._models: list[ModelInfo] = []
        self._load()

    def _load(self) -> None:
        self._models.clear()
        root = self._config.catalog_root
        if not root.exists():
            LOGGER.warning("Catalog root %s does not exist", root)
            return
        for meta_path in root.glob("**/meta.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to read metadata %s: %s", meta_path, exc)
                continue
            relative = meta_path.relative_to(root)
            gender = relative.parts[0] if relative.parts else "unknown"
            model_id = str(relative.parent)
            title = data.get("title") or relative.parent.name
            product_url = data.get("product_url", "")
            self._models.append(
                ModelInfo(
                    model_id=model_id,
                    title=title,
                    product_url=product_url,
                    gender=gender,
                    metadata={k: str(v) for k, v in data.items()},
                )
            )

    def pick_models(self, gender: str, count: int) -> list[ModelInfo]:
        pool = [m for m in self._models if m.gender in {gender, "unisex"}]
        if not pool:
            pool = list(self._models)
        if not pool:
            raise RuntimeError("Catalog is empty")
        if len(pool) >= count:
            return random.sample(pool, count)
        choices = pool.copy()
        while len(choices) < count:
            choices.append(random.choice(pool))
        return choices[:count]


class TryOnGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config
        self._catalog = ModelCatalog(config)
        self._font = ImageFont.load_default()

    def refresh_catalog(self) -> None:
        self._catalog._load()

    async def generate(self, gender: str) -> CollageResult:
        models = self._catalog.pick_models(gender, 2)
        image = await self._build_collage(models)
        return CollageResult(image=image, models=models)

    async def _build_collage(self, models: Iterable[ModelInfo]) -> io.BytesIO:
        cfg = self._config
        base = Image.new("RGB", (cfg.collage_width, cfg.collage_height), cfg.background_color)
        draw = ImageDraw.Draw(base)
        tile_width = (cfg.collage_width - cfg.margin * 2 - cfg.divider_width) // 2
        tile_height = cfg.collage_height - cfg.margin * 2
        offset_x = cfg.margin
        colors = [
            (234, 240, 255),
            (255, 240, 245),
            (240, 255, 250),
            (255, 253, 231),
            (245, 245, 255),
        ]
        for idx, model in enumerate(models):
            tile = Image.new("RGB", (tile_width, tile_height), random.choice(colors))
            tile_draw = ImageDraw.Draw(tile)
            text = f"{model.title}\n{model.metadata.get('shape', '')}"
            self._draw_centered_text(tile_draw, tile_width, tile_height, text)
            base.paste(tile, (offset_x, cfg.margin))
            offset_x += tile_width + cfg.divider_width
            if idx == 0:
                draw.rectangle(
                    [
                        (offset_x - cfg.divider_width, cfg.margin),
                        (offset_x - 1, cfg.margin + tile_height),
                    ],
                    fill=cfg.divider_color,
                )
        output = io.BytesIO()
        base.save(output, format="JPEG", quality=88)
        output.seek(0)
        return output

    def _draw_centered_text(self, draw: ImageDraw.ImageDraw, width: int, height: int, text: str) -> None:
        lines = text.split("\n")
        line_height = self._font.getbbox("Ag")[3]
        total_height = len(lines) * line_height + (len(lines) - 1) * 8
        y = (height - total_height) // 2
        for line in lines:
            bbox = self._font.getbbox(line)
            line_width = bbox[2]
            x = (width - line_width) // 2
            draw.text((x, y), line, fill=(40, 40, 40), font=self._font)
            y += line_height + 8


__all__ = ["GeneratorConfig", "ModelInfo", "CollageResult", "TryOnGenerator"]
