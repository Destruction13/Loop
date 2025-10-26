from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class TryOnStates(StatesGroup):
    waiting_for_gender = State()
    waiting_for_photo = State()
    viewing_results = State()
