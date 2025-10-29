"""Helpers for downloading Google Drive hosted frame references."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlparse

import aiohttp
from PIL import Image

from app.utils.paths import ensure_dir

_DRIVE_FILE_RE: Final = re.compile(r"/d/([^/]+)/")
_TIMEOUT = aiohttp.ClientTimeout(total=60)


def extract_drive_id(url: str) -> str:
    """Return the file id from a Drive sharing URL."""

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError("Drive URL must contain a host")
    if "/file/d/" in parsed.path:
        match = _DRIVE_FILE_RE.search(parsed.path)
        if match:
            return match.group(1)
    query = parse_qs(parsed.query)
    ids = query.get("id")
    if ids:
        return ids[0]
    raise ValueError("Unsupported Google Drive URL format")


async def fetch_drive_file(url: str, cache_dir: str = ".cache/frames") -> str:
    """Download the file pointed by a Drive URL and cache it locally as PNG."""

    drive_id = extract_drive_id(url)
    cache_path = ensure_dir(Path(cache_dir)) / f"{drive_id}.png"
    if cache_path.exists():
        return str(cache_path)

    direct_url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.get(direct_url) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Failed to fetch Drive file {drive_id}: status {response.status}"
                )
            content = await response.read()
    image = Image.open(io.BytesIO(content))
    image.load()
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA")
    image.save(cache_path, format="PNG")
    return str(cache_path)


__all__ = ["extract_drive_id", "fetch_drive_file"]

