from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import datetime
from typing import Optional

from app import messages_ru as msg
from app.fsm import (
    GenerationOutcome,
    next_first_flag_value,
    resolve_generation_followup,
)
from app.keyboards import (
    attach_photo_keyboard,
    limit_reached_keyboard,
    promo_keyboard,
    reminder_keyboard,
)
from app.models import UserProfile, FilterOptions
from app.services.scheduler import ReminderScheduler


def test_filter_options_only_gender() -> None:
    field_names = [field.name for field in fields(FilterOptions)]

    assert field_names == ["gender"]


def test_photo_instruction_text_and_keyboard() -> None:
    expected_text = (
        "Кинь нам селфи или любую чёткую фотку, где видно лицо прямо. "
        "Можно взять из “Избранного” — не обязательно себя сейчас фоткать. "
        "Главное — лицо в кадре"
    )
    assert msg.PHOTO_INSTRUCTIONS == expected_text

    keyboard = attach_photo_keyboard()
    assert keyboard.inline_keyboard[0][0].text == msg.ATTACH_PHOTO_BUTTON


def test_followup_resolution_variants() -> None:
    first_plan = resolve_generation_followup(first_generated_today=True, remaining=5)
    assert first_plan.outcome is GenerationOutcome.FIRST
    assert first_plan.remaining == 5

    follow_plan = resolve_generation_followup(first_generated_today=False, remaining=4)
    assert follow_plan.outcome is GenerationOutcome.FOLLOWUP

    limit_plan = resolve_generation_followup(first_generated_today=True, remaining=0)
    assert limit_plan.outcome is GenerationOutcome.LIMIT
    assert limit_plan.remaining == 0


def test_limit_flow_keyboards() -> None:
    landing = "https://example.com/booking"
    limit_keyboard = limit_reached_keyboard(landing)

    assert [len(row) for row in limit_keyboard.inline_keyboard] == [1, 1, 1]
    assert limit_keyboard.inline_keyboard[0][0].url == landing
    assert limit_keyboard.inline_keyboard[1][0].callback_data == "limit_promo"
    assert limit_keyboard.inline_keyboard[2][0].callback_data == "limit_remind"

    promo = promo_keyboard(landing)
    assert [len(row) for row in promo.inline_keyboard] == [1, 1]
    assert promo.inline_keyboard[0][0].url == landing
    assert promo.inline_keyboard[1][0].callback_data == "limit_remind"


def test_reminder_scheduler_sends_message_with_keyboard() -> None:
    class DummyBot:
        def __init__(self) -> None:
            self.sent: list[tuple[int, str, object]] = []

        async def send_message(self, chat_id: int, text: str, reply_markup=None) -> None:  # noqa: ANN001
            self.sent.append((chat_id, text, reply_markup))

    class DummyRepository:
        def __init__(self) -> None:
            self.cleared: list[tuple[int, Optional[datetime]]] = []
            self.profiles = [UserProfile(user_id=42)]

        async def list_due_reminders(self, now: datetime) -> list[UserProfile]:
            return self.profiles

        async def set_reminder(self, user_id: int, when: Optional[datetime]) -> None:
            self.cleared.append((user_id, when))

    async def scenario() -> None:
        bot = DummyBot()
        repo = DummyRepository()
        scheduler = ReminderScheduler(
            bot=bot,
            repository=repo,
            message_text=msg.REMINDER_MESSAGE,
            keyboard_factory=reminder_keyboard,
            interval_seconds=0,
        )
        await scheduler._check_and_send()  # noqa: SLF001

        assert repo.cleared == [(42, None)]
        assert bot.sent[0][0] == 42
        assert bot.sent[0][1] == msg.REMINDER_MESSAGE
        keyboard = bot.sent[0][2]
        assert keyboard.inline_keyboard[0][0].text == msg.REMINDER_PROMPT_BUTTON

    asyncio.run(scenario())


def test_first_flag_persistence_rules() -> None:
    assert next_first_flag_value(False, GenerationOutcome.FOLLOWUP) is False
    assert next_first_flag_value(True, GenerationOutcome.LIMIT) is False
