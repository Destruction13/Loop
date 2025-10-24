"""Utility helpers for path management."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str) -> str:
    """Basic sanitization for filenames."""

    return "".join(ch for ch in filename if ch.isalnum() or ch in {"_", "-", "."})
