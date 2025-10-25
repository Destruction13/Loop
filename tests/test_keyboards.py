from __future__ import annotations

from app.keyboards import generation_result_keyboard, pair_selection_keyboard


def test_pair_selection_keyboard_contains_expected_buttons() -> None:
    keyboard = pair_selection_keyboard(
        [
            (1, "id1", "Alpha"),
            (2, "id2", "Beta"),
        ],
        index_style="ascii",
        max_title_length=24,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 2
    first_button = row[0]
    second_button = row[1]
    assert first_button.text == "[1] Выбрать: Alpha"
    assert first_button.callback_data == "pick|id1"
    assert second_button.text == "[2] Выбрать: Beta"
    assert second_button.callback_data == "pick|id2"


def test_pair_selection_keyboard_emoji_style() -> None:
    keyboard = pair_selection_keyboard(
        [
            (3, "id3", "Gamma"),
            (4, "id4", "Delta"),
        ],
        index_style="emoji",
        max_title_length=24,
    )

    row = keyboard.inline_keyboard[0]
    assert row[0].text.startswith("3️⃣")
    assert row[1].text.startswith("4️⃣")


def test_pair_selection_keyboard_none_style() -> None:
    keyboard = pair_selection_keyboard(
        [
            (1, "id1", "Alpha"),
            (2, "id2", "Beta"),
        ],
        index_style="none",
        max_title_length=24,
    )

    row = keyboard.inline_keyboard[0]
    assert row[0].text == "Выбрать: Alpha"
    assert row[1].text == "Выбрать: Beta"


def test_pair_selection_keyboard_truncates_titles() -> None:
    keyboard = pair_selection_keyboard(
        [(1, "id1", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")],
        index_style="ascii",
        max_title_length=10,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 1
    assert row[0].text == "[1] Выбрать: ABCDEFGHI…"


def test_pair_selection_keyboard_single_model_uses_index() -> None:
    keyboard = pair_selection_keyboard(
        [(2, "id1", "Solo")],
        index_style="ascii",
        max_title_length=24,
    )

    row = keyboard.inline_keyboard[0]
    assert row[0].text == "[2] Выбрать: Solo"


def test_generation_result_keyboard_with_remaining_options() -> None:
    keyboard = generation_result_keyboard("https://example.com", 3)

    assert len(keyboard.inline_keyboard) == 1
    buttons = keyboard.inline_keyboard[0]
    assert len(buttons) == 2
    assert buttons[0].url == "https://example.com"
    assert buttons[1].callback_data == "more|3"


def test_generation_result_keyboard_without_remaining_options() -> None:
    keyboard = generation_result_keyboard("https://example.com", 0)

    assert len(keyboard.inline_keyboard) == 1
    buttons = keyboard.inline_keyboard[0]
    assert len(buttons) == 1
    assert buttons[0].url == "https://example.com"
