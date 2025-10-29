"""Image IO helpers used in the generation pipeline."""

from __future__ import annotations

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


__all__ = ["save_user_photo", "redownload_user_photo", "resize_inplace"]

