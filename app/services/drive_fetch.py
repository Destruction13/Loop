"""Helpers for downloading Google Drive hosted frame references."""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlparse

import aiohttp
from PIL import Image

from app.utils.paths import ensure_dir

_DRIVE_FILE_RE: Final = re.compile(r"/d/([^/]+)/")
_CONFIRM_RE: Final = re.compile(r"confirm=([0-9A-Za-z_]+)")
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


@dataclass(slots=True)
class DriveDownload:
    """Downloaded Drive file content with derived metadata."""

    data: bytes
    mime: str
    extension: str
    size: int
    drive_id: str


async def fetch_drive_bytes(url: str, *, retries: int = 3) -> DriveDownload:
    """Download a Drive file into memory and validate it as an image."""

    drive_id = extract_drive_id(url)
    direct_url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    attempt = 0
    last_exc: Exception | None = None
    while attempt < max(retries, 1):
        attempt += 1
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
                data, content_type, status = await _download_drive_bytes(
                    session, direct_url
                )
            if status != 200:
                raise RuntimeError(
                    f"Failed to fetch Drive file {drive_id}: status {status}"
                )
            mime, ext = _resolve_image_mime(data, content_type)
            return DriveDownload(
                data=data,
                mime=mime,
                extension=ext,
                size=len(data),
                drive_id=drive_id,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max(retries, 1):
                break
            await asyncio.sleep(0.5 * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch Drive file {drive_id}")


async def _download_drive_bytes(
    session: aiohttp.ClientSession, url: str
) -> tuple[bytes, str, int]:
    async with session.get(url, allow_redirects=True) as response:
        status = response.status
        content_type = (response.headers.get("Content-Type") or "").strip()
        data = await response.read()
    if content_type.lower().startswith("text/html") or _looks_like_html(data):
        confirm = _extract_confirm_token(data)
        if confirm:
            confirm_url = f"{url}&confirm={confirm}"
            async with session.get(confirm_url, allow_redirects=True) as response:
                status = response.status
                content_type = (response.headers.get("Content-Type") or "").strip()
                data = await response.read()
        else:
            return data, content_type, status
    return data, content_type, status


def _extract_confirm_token(payload: bytes) -> str | None:
    text = payload.decode("utf-8", errors="ignore")
    match = _CONFIRM_RE.search(text)
    if match:
        return match.group(1)
    return None


def _looks_like_html(payload: bytes) -> bool:
    snippet = payload[:200].lstrip().lower()
    return snippet.startswith(b"<!doctype html") or snippet.startswith(b"<html")


def _resolve_image_mime(data: bytes, content_type: str) -> tuple[str, str]:
    if not data:
        raise ValueError("Drive file is empty")
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized.startswith("image/"):
        ext = normalized.split("/", 1)[1].strip() or "png"
        ext = "jpg" if ext == "jpeg" else ext
        return normalized, ext
    if normalized and normalized not in {"application/octet-stream"}:
        raise ValueError(f"Drive file is not an image (mime={normalized})")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        format_hint = (image.format or "PNG").upper()
    mime = Image.MIME.get(format_hint, "image/png")
    ext = format_hint.lower()
    ext = "jpg" if ext == "jpeg" else ext
    if not mime.startswith("image/"):
        raise ValueError("Drive file is not an image")
    return mime, ext


__all__ = ["extract_drive_id", "fetch_drive_file", "fetch_drive_bytes", "DriveDownload"]

