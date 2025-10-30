from __future__ import annotations

from app.fsm import chunk_models
from app.models import GlassModel


def _model(uid: str, title: str) -> GlassModel:
    return GlassModel(
        unique_id=uid,
        title=title,
        model_code="code",
        site_url="https://example.com",
        img_user_url=f"https://img/{uid}.jpg",
        img_nano_url=f"https://nano/{uid}.jpg",
        gender="unisex",
    )


def test_chunk_models_splits_into_batches() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 7)]

    batches = chunk_models(models, 3)

    assert len(batches) == 2
    assert batches[0] == (models[0], models[1], models[2])
    assert batches[1] == (models[3], models[4], models[5])


def test_chunk_models_handles_short_last_batch() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 5)]

    batches = chunk_models(models, 3)

    assert len(batches) == 2
    assert batches[0] == (models[0], models[1], models[2])
    assert batches[1] == (models[3],)
