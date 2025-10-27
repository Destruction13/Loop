from __future__ import annotations

from app.keyboards import (
    batch_selection_keyboard,
    generation_result_keyboard,
    main_reply_keyboard,
)
from app.texts import messages as msg


def test_batch_selection_keyboard_contains_expected_buttons() -> None:
    keyboard = batch_selection_keyboard(
        [
            ("id1", "Garnet Black"),
            ("id2", "Antimony Grey"),
        ],
        source="batch3",
        max_title_length=28,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 2
    assert row[0].text == "Garnet Black"
    assert row[0].callback_data == "pick:batch3:id1"
    assert row[1].text == "Antimony Grey"
    assert row[1].callback_data == "pick:batch3:id2"


def test_batch_selection_keyboard_three_buttons_single_row() -> None:
    keyboard = batch_selection_keyboard(
        [
            ("id1", "Garnet Black"),
            ("id2", "Antimony Grey"),
            ("id3", "Quartz Silver"),
        ],
        source="batch3",
        max_title_length=28,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 3
    assert [button.callback_data for button in row] == [
        "pick:batch3:id1",
        "pick:batch3:id2",
        "pick:batch3:id3",
    ]


def test_batch_selection_keyboard_truncates_titles() -> None:
    keyboard = batch_selection_keyboard(
        [("id1", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")],
        source="batch3",
        max_title_length=10,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 1
    assert row[0].text == "ABCDEFGHI…"


def test_batch_selection_keyboard_single_model() -> None:
    keyboard = batch_selection_keyboard(
        [("id1", "Solo Model")],
        source="batch3",
        max_title_length=28,
    )

    assert len(keyboard.inline_keyboard) == 1
    row = keyboard.inline_keyboard[0]
    assert len(row) == 1
    assert row[0].text == "Solo Model"
    assert row[0].callback_data == "pick:batch3:id1"


def test_generation_result_keyboard_with_remaining_options() -> None:
    keyboard = generation_result_keyboard("https://example.com", 3)

    assert len(keyboard.inline_keyboard) == 2
    details_row = keyboard.inline_keyboard[0]
    more_row = keyboard.inline_keyboard[1]
    assert [button.text for button in details_row] == [msg.DETAILS_BUTTON_TEXT]
    assert [button.text for button in more_row] == [
        f"{msg.MORE_VARIANTS_BUTTON_TEXT} (осталось 3)"
    ]
    assert details_row[0].url == "https://example.com"
    assert more_row[0].callback_data == "more|3"


def test_generation_result_keyboard_without_remaining_options() -> None:
    keyboard = generation_result_keyboard("https://example.com", 0)

    assert len(keyboard.inline_keyboard) == 1
    buttons = keyboard.inline_keyboard[0]
    assert len(buttons) == 1
    assert buttons[0].url == "https://example.com"


def test_main_reply_keyboard_layout() -> None:
    keyboard = main_reply_keyboard()

    assert keyboard.resize_keyboard is True
    assert keyboard.one_time_keyboard is False
    assert [[button.text for button in row] for row in keyboard.keyboard] == [
        [msg.MAIN_MENU_TRY_BUTTON],
        [msg.MAIN_MENU_POLICY_BUTTON],
    ]
