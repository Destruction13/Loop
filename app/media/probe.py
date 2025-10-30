"""Video probing helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ProbeResult:
    width: int | None
    height: int | None
    rotation: int
    source: str
    raw: dict[str, Any]


def _safe_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_rotation(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sidecar_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.meta.json")


def _load_cache(video_path: Path, stat: os.stat_result) -> _ProbeResult | None:
    sidecar = _sidecar_path(video_path)
    if not sidecar.exists():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as fp:
            cached = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read cached metadata from %s: %s", sidecar, exc)
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("src_mtime") != stat.st_mtime or cached.get("src_size") != stat.st_size:
        return None
    width = _safe_positive_int(cached.get("width"))
    height = _safe_positive_int(cached.get("height"))
    rotation = _safe_rotation(cached.get("rotation"))
    if not width or not height:
        return None
    return _ProbeResult(
        width=width,
        height=height,
        rotation=rotation,
        source="cache",
        raw=cached,
    )


def _store_cache(video_path: Path, stat: os.stat_result, result: _ProbeResult) -> None:
    sidecar = _sidecar_path(video_path)
    payload = {
        "src_mtime": stat.st_mtime,
        "src_size": stat.st_size,
        "width": result.width,
        "height": result.height,
        "rotation": result.rotation,
        "probed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to write metadata cache %s: %s", sidecar, exc)


def _apply_rotation(result: _ProbeResult) -> Tuple[int | None, int | None]:
    width, height = result.width, result.height
    if width and height:
        rotation = result.rotation % 360
        if rotation in {90, 270}:
            return height, width
    return width, height


def _probe_with_ffprobe(video_path: Path) -> _ProbeResult | None:
    executable = shutil.which("ffprobe")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,rotation,side_data_list",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffprobe failed for %s: %s", video_path, exc)
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("ffprobe output parse error for %s: %s", video_path, exc)
        return None
    streams = payload.get("streams")
    if not streams:
        return None
    stream = streams[0] or {}
    width = _safe_positive_int(stream.get("width"))
    height = _safe_positive_int(stream.get("height"))
    rotation_value = stream.get("rotation")
    if rotation_value is None:
        side_data = stream.get("side_data_list") or []
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                rotation_value = entry.get("rotation")
                break
    rotation = _safe_rotation(rotation_value)
    return _ProbeResult(
        width=width,
        height=height,
        rotation=rotation,
        source="ffprobe",
        raw=payload,
    )


def _probe_with_opencv(video_path: Path) -> _ProbeResult | None:
    try:
        import cv2  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    capture = cv2.VideoCapture(str(video_path))  # type: ignore[attr-defined]
    try:
        if not capture.isOpened():
            return None
        width = _safe_positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[attr-defined]
        height = _safe_positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[attr-defined]
    finally:
        capture.release()
    return _ProbeResult(
        width=width,
        height=height,
        rotation=0,
        source="opencv",
        raw={},
    )


def probe_video_size(path: str) -> tuple[int | None, int | None, dict[str, Any]]:
    """Probe the video dimensions using available tools."""

    video_path = Path(path)
    meta: dict[str, Any] = {"source": "none"}
    try:
        stat = video_path.stat()
    except FileNotFoundError:
        meta.update({"source": "missing"})
        return None, None, meta
    except OSError as exc:
        meta.update({"source": "stat_error", "error": str(exc)})
        return None, None, meta

    cache_hit = _load_cache(video_path, stat)
    if cache_hit and cache_hit.width and cache_hit.height:
        width, height = _apply_rotation(cache_hit)
        meta.update({"source": cache_hit.source, "rotation": cache_hit.rotation, "cache": cache_hit.raw})
        return width, height, meta

    for probe in (_probe_with_ffprobe, _probe_with_opencv):
        result = probe(video_path)
        if not result or not result.width or not result.height:
            continue
        width, height = _apply_rotation(result)
        meta.update({"source": result.source, "rotation": result.rotation})
        _store_cache(video_path, stat, result)
        return width, height, meta

    meta.update({"source": "none"})
    return None, None, meta
