from __future__ import annotations

from app.keyboards import generation_result_keyboard, pair_selection_keyboard


def test_pair_selection_keyboard_contains_expected_buttons() -> None:
    keyboard = pair_selection_keyboard(
        [
            ("id1", "Garnet Black"),
            ("id2", "Antimony Grey"),
        ],
        max_title_length=28,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 2
    assert row[0].text == "Garnet Black"
    assert row[0].callback_data == "pick|id1"
    assert row[1].text == "Antimony Grey"
    assert row[1].callback_data == "pick|id2"


def test_pair_selection_keyboard_truncates_titles() -> None:
    keyboard = pair_selection_keyboard(
        [("id1", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")],
        max_title_length=10,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 1
    assert row[0].text == "ABCDEFGHI…"


def test_pair_selection_keyboard_single_model() -> None:
    keyboard = pair_selection_keyboard(
        [("id1", "Solo Model")],
        max_title_length=28,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 1
    assert row[0].text == "Solo Model"
    assert row[0].callback_data == "pick|id1"


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
