"""Helpers for working with Google Drive URLs."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

__all__ = ["drive_view_to_direct", "DriveUrlError", "DriveFolderUrlError"]


class DriveUrlError(ValueError):
    """Base exception for invalid Google Drive URLs."""


class DriveFolderUrlError(DriveUrlError):
    """Raised when the provided URL points to a folder, not a file."""


_FILE_ID_PATTERN = re.compile(r"/d/([a-zA-Z0-9_-]+)")
_FRAGMENT_ID_PATTERN = re.compile(r"id=([a-zA-Z0-9_-]+)")
_INVISIBLE_CHARS = "\u200b\u200c\u200d\ufeff"


def drive_view_to_direct(url: str, export: str = "view") -> str:
    """Convert a Google Drive share link to a direct view URL.

    Args:
        url: Original Google Drive sharing link.
        export: Export parameter for the resulting link (default: ``view``).

    Returns:
        Direct link suitable for Telegram.

    Raises:
        DriveFolderUrlError: If the URL points to a folder.
        DriveUrlError: If the URL does not contain a Drive file identifier.
    """

    cleaned = (url or "").strip().strip(_INVISIBLE_CHARS)
    if not cleaned:
        raise DriveUrlError("Empty Drive URL")

    parsed = urlparse(cleaned)
    path = parsed.path
    fragment = parsed.fragment
    path_lower = path.lower()
    fragment_lower = fragment.lower()
    query = parse_qs(parsed.query)

    if (
        "/folders/" in path_lower
        or "folderview" in path_lower
        or "folders" in fragment_lower
        or (path_lower.startswith("/drive/") and "id" in query)
    ):
        raise DriveFolderUrlError("Drive URL points to a folder")

    file_id = _extract_file_id(path, query, fragment)
    if not file_id:
        raise DriveUrlError("Could not extract file id from Drive URL")

    return f"https://drive.google.com/uc?export={export}&id={file_id}"


def _extract_file_id(path: str, query: dict[str, list[str]], fragment: str) -> str | None:
    """Extract the Drive file identifier from various URL formats."""

    # Pattern /file/d/<ID>/view or /d/<ID>
    match = _FILE_ID_PATTERN.search(path)
    if match:
        return match.group(1)

    # Query parameter id=<ID> (open?id=..., uc?id=..., etc.)
    if "id" in query and query["id"]:
        candidate = query["id"][0].strip()
        if candidate:
            return candidate

    # Some share links keep id in fragment (e.g. ...#id=<ID>)
    frag_match = _FRAGMENT_ID_PATTERN.search(fragment)
    if frag_match and "folders" not in fragment.lower():
        return frag_match.group(1)

    return None
