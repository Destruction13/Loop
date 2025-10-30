import json
import subprocess

from app.media.probe import probe_video_size


def test_probe_video_size_ffprobe_rotation(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    payload = {
        "streams": [
            {
                "width": 720,
                "height": 1280,
                "rotation": 90,
            }
        ]
    }

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffprobe")

    def fake_run(cmd, check, capture_output, text):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    width, height, meta = probe_video_size(str(video_path))

    assert width == 1280
    assert height == 720
    assert meta["source"] == "ffprobe"
    assert meta["rotation"] == 90

    cache_path = video_path.with_name("clip.meta.json")
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text())
    assert cached["width"] == 720
    assert cached["height"] == 1280
    assert cached["rotation"] == 90


def test_probe_video_size_cache_behavior(tmp_path, monkeypatch):
    video_path = tmp_path / "cache.mp4"
    video_path.write_bytes(b"video")
    stat = video_path.stat()
    cache_path = video_path.with_name("cache.meta.json")
    cache_payload = {
        "src_mtime": stat.st_mtime,
        "src_size": stat.st_size,
        "width": 640,
        "height": 360,
        "rotation": 270,
        "probed_at": "2024-01-01T00:00:00+00:00",
    }
    cache_path.write_text(json.dumps(cache_payload))

    import app.media.probe as probe_module

    calls = {"ffprobe": 0, "opencv": 0}

    def fake_ffprobe(_path):
        calls["ffprobe"] += 1
        return None

    def fake_opencv(_path):
        calls["opencv"] += 1
        return None

    monkeypatch.setattr(probe_module, "_probe_with_ffprobe", fake_ffprobe)
    monkeypatch.setattr(probe_module, "_probe_with_opencv", fake_opencv)

    width, height, meta = probe_video_size(str(video_path))
    assert width == 360
    assert height == 640
    assert meta["source"] == "cache"
    assert calls == {"ffprobe": 0, "opencv": 0}

    video_path.write_bytes(b"changed")

    width2, height2, meta2 = probe_video_size(str(video_path))
    assert width2 is None
    assert height2 is None
    assert meta2["source"] == "none"
    assert calls == {"ffprobe": 1, "opencv": 1}
