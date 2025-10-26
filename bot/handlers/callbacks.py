from __future__ import annotations

import random
from dataclasses import dataclass

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

import messages
from bot import keyboards
from bot.idle import IdleWatcher
from bot.states import TryOnStates
from db.init import Database


@dataclass(slots=True)
class CallbackConfig:
    delete_old_message_on_select: bool


def register_callback_handlers(
    router: Router,
    db: Database,
    idle: IdleWatcher,
    config: CallbackConfig,
) -> None:
    @router.callback_query(keyboards.ModelSelectCallback.filter())
    async def handle_model_select(
        call: CallbackQuery,
        callback_data: keyboards.ModelSelectCallback,
        state: FSMContext,
    ) -> None:
        await call.answer()
        tg_id = call.from_user.id
        await idle.touch(tg_id)
        data = await state.get_data()
        models = data.get("models", [])
        model = next((m for m in models if m.get("model_id") == callback_data.model_id), None)
        if model is None:
            await call.answer(messages.NEW_PHOTO_REQUIRED_TEXT, show_alert=True)
            return
        if call.message:
            try:
                await call.message.edit_reply_markup()
            except Exception:
                pass
            if config.delete_old_message_on_select:
                try:
                    await call.message.delete()
                except Exception:
                    pass
        await db.log_event(
            tg_id,
            "model_selected",
            {"model_id": model.get("model_id"), "title": model.get("title")},
        )
        text = "\n\n".join(
            [
                messages.MODEL_SELECTION_PROMPT.format(title=model.get("title", "")),
                messages.MORE_VARIANTS_PROMPT,
            ]
        )
        reply_markup = keyboards.selected_model_keyboard(model.get("product_url", ""))
        if call.message:
            await call.message.answer(text, reply_markup=reply_markup)
        else:
            await call.bot.send_message(tg_id, text, reply_markup=reply_markup)
        await state.update_data(selected_model=model)
        await state.set_state(TryOnStates.waiting_for_photo)

    @router.callback_query(keyboards.MoreOptionsCallback.filter())
    async def handle_more_options(call: CallbackQuery, state: FSMContext) -> None:
        await call.answer()
        tg_id = call.from_user.id
        await idle.touch(tg_id)
        if call.message:
            try:
                await call.message.edit_reply_markup()
            except Exception:
                pass
        response_text = random.choice(list(messages.NEXT_RESULT_CAPTIONS))
        if call.message:
            await call.message.answer(response_text)
        else:
            await call.bot.send_message(tg_id, response_text)
        await state.set_state(TryOnStates.waiting_for_photo)
