"""Tests for custom inline keyboards."""

from __future__ import annotations

from app.keyboards import generation_result_keyboard, pair_selection_keyboard


def test_pair_selection_keyboard_contains_expected_buttons() -> None:
    keyboard = pair_selection_keyboard(
        [
            ("id1", "Alpha"),
            ("id2", "Beta"),
        ],
        max_title_length=24,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 2
    first_button = row[0]
    second_button = row[1]
    assert first_button.text == "① Выбрать: Alpha"
    assert first_button.callback_data == "pick|id1"
    assert second_button.text == "② Выбрать: Beta"
    assert second_button.callback_data == "pick|id2"


def test_pair_selection_keyboard_truncates_titles() -> None:
    keyboard = pair_selection_keyboard(
        [("id1", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")],
        max_title_length=10,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 1
    assert row[0].text == "① Выбрать: ABCDEFGHI…"


def test_pair_selection_keyboard_single_model_uses_ordinal() -> None:
    keyboard = pair_selection_keyboard(
        [("id1", "Solo")],
        start_index=2,
        max_title_length=24,
    )

    row = keyboard.inline_keyboard[0]
    assert row[0].text == "② Выбрать: Solo"


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
