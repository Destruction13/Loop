"""Tests for pick utilities."""

from __future__ import annotations

from pathlib import Path

from app.models import ModelItem, ModelMeta
from app.utils.pick import pick_four


def make_model(model_id: str, shape: str) -> ModelItem:
    return ModelItem(
        model_id=model_id,
        thumb_path=Path("thumb.png"),
        overlay_path=None,
        meta=ModelMeta(title=model_id, product_url="https://example.com", shape=shape),
    )


def test_pick_prioritizes_unseen() -> None:
    models = [
        make_model("m1", "round"),
        make_model("m2", "square"),
        make_model("m3", "round"),
        make_model("m4", "oval"),
        make_model("m5", "square"),
    ]
    picked = pick_four(models, seen=["m1", "m2"], limit=4)
    assert all(model.model_id not in {"m1", "m2"} for model in picked[:2])
    assert len(picked) == 4
