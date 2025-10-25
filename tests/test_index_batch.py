from app.fsm import build_indexed_batch, pair_indexed_models
from app.keyboards import pair_selection_keyboard
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


def test_build_indexed_batch_assigns_incremental_indices() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 5)]
    batch = build_indexed_batch(models)

    assert [item.index for item in batch] == [1, 2, 3, 4]
    assert [item.model.unique_id for item in batch] == [f"id{i}" for i in range(1, 5)]


def test_pair_indexed_models_produces_expected_pairs() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 5)]
    batch = build_indexed_batch(models)
    pairs = pair_indexed_models(batch)

    assert len(pairs) == 2
    assert [item.index for item in pairs[0]] == [1, 2]
    assert [item.index for item in pairs[1]] == [3, 4]


def test_keyboard_labels_follow_batch_indices() -> None:
    models = [_model(f"id{i}", f"Title {i}") for i in range(1, 5)]
    batch = build_indexed_batch(models)
    pairs = pair_indexed_models(batch)

    first_keyboard = pair_selection_keyboard(
        [(item.index, item.model.unique_id, item.model.title) for item in pairs[0]],
        index_style="ascii",
        max_title_length=24,
    )
    second_keyboard = pair_selection_keyboard(
        [(item.index, item.model.unique_id, item.model.title) for item in pairs[1]],
        index_style="ascii",
        max_title_length=24,
    )

    first_row = first_keyboard.inline_keyboard[0]
    second_row = second_keyboard.inline_keyboard[0]
    assert first_row[0].text.startswith("[1]")
    assert first_row[1].text.startswith("[2]")
    assert second_row[0].text.startswith("[3]")
    assert second_row[1].text.startswith("[4]")
