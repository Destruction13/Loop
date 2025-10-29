"""Image IO helpers used in the generation pipeline."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aiogram import Bot
from aiogram.types import Message
from PIL import Image, ImageOps

from app.utils.paths import ensure_dir


async def save_user_photo(message: Message, tmp_dir: str = "tmp") -> str:
    """Download the incoming photo message into a temporary directory."""

    if not message.photo:
        raise ValueError("Message does not contain a photo")
    user_id = message.from_user.id if message.from_user else 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    filename = f"user_{user_id}_{timestamp}.jpg"
    target_dir = ensure_dir(Path(tmp_dir))
    destination = target_dir / filename
    await message.bot.download(message.photo[-1], destination=destination)
    return str(destination)


async def redownload_user_photo(
    bot: Bot, file_id: str, user_id: int, tmp_dir: str = "tmp"
) -> str:
    """Download a previously uploaded photo by Telegram file_id."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    filename = f"user_{user_id}_{timestamp}.jpg"
    target_dir = ensure_dir(Path(tmp_dir))
    destination = target_dir / filename
    await bot.download(file_id, destination=destination)
    return str(destination)


def resize_inplace(path: str | Path, max_side: int = 2048) -> None:
    """Resize an image in-place so that the longest side is at most max_side."""

    file_path = Path(path)
    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.LANCZOS)
        format_hint = (image.format or "").upper()
        if not format_hint:
            format_hint = "PNG" if file_path.suffix.lower() == ".png" else "JPEG"
        save_kwargs: dict[str, Optional[int]] = {}
        if format_hint == "JPEG" and image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        if format_hint == "JPEG":
            save_kwargs["quality"] = 95
        image.save(file_path, format=format_hint, **save_kwargs)


def load_image_metadata(path: str | Path) -> tuple[tuple[int, int], bytes | None]:
    """Return (width, height) and optional EXIF blob for the provided image."""

    file_path = Path(path)
    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)
        size = image.size
        exif_bytes = image.info.get("exif")
    return size, exif_bytes


def ensure_dimensions(
    image_bytes: bytes,
    size: tuple[int, int],
    *,
    exif: bytes | None = None,
) -> bytes:
    """Resize the in-memory image to the exact size, preserving format and optional EXIF."""

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        target_size = tuple(size)
        if image.size != target_size:
            image = image.resize(target_size, Image.LANCZOS)
        output = io.BytesIO()
        format_hint = (image.format or "PNG").upper()
        save_kwargs: dict[str, bytes | int] = {}
        if format_hint == "JPEG" and image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        if format_hint == "JPEG":
            if exif:
                save_kwargs["exif"] = exif
            save_kwargs["quality"] = 95
        elif format_hint in {"TIFF"} and exif:
            save_kwargs["exif"] = exif
        image.save(output, format=format_hint, **save_kwargs)
        return output.getvalue()


__all__ = [
    "save_user_photo",
    "redownload_user_photo",
    "resize_inplace",
    "load_image_metadata",
    "ensure_dimensions",
]

