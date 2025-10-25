"""Helpers for working with Google Drive URLs."""

from __future__ import annotations

import re

__all__ = ["drive_view_to_direct"]


_DRIVE_ID_PATTERN = re.compile(r"/d/([a-zA-Z0-9_-]+)/")


def drive_view_to_direct(url: str, export: str = "view") -> str:
    """Convert a Google Drive view link to a direct download/view link.

    Args:
        url: Original Google Drive sharing link.
        export: Export parameter for the resulting link (default: ``view``).

    Returns:
        Direct link suitable for Telegram.

    Raises:
        ValueError: If the URL does not contain a Drive file identifier.
    """

    match = _DRIVE_ID_PATTERN.search(url)
    if not match:
        raise ValueError("Could not extract file id from Drive URL")
    file_id = match.group(1)
    return f"https://drive.google.com/uc?export={export}&id={file_id}"
