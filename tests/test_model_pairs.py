from __future__ import annotations

from app.fsm import pair_models
from app.models import GlassModel


def _model(uid: str, title: str) -> GlassModel:
    return GlassModel(
        unique_id=uid,
        title=title,
        model_code="code",
        site_url="https://example.com",
        img_user_url=f"https://img/{uid}.jpg",
        img_nano_url=f"https://nano/{uid}.jpg",
        gender="Унисекс",
    )


def test_pair_models_splits_into_pairs() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 5)]

    pairs = pair_models(models)

    assert len(pairs) == 2
    assert pairs[0] == (models[0], models[1])
    assert pairs[1] == (models[2], models[3])


def test_pair_models_handles_odd_count() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 4)]

    pairs = pair_models(models)

    assert len(pairs) == 2
    assert pairs[0] == (models[0], models[1])
    assert pairs[1] == (models[2],)
