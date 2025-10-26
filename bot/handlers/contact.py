from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message

import messages
from bot import keyboards
from bot.idle import IdleWatcher
from db.init import Database


def register_contact_handler(router: Router, db: Database, idle: IdleWatcher) -> None:
    @router.message(F.contact)
    async def handle_contact(message: Message) -> None:
        if not message.contact:
            return
        tg_id = message.from_user.id
        await idle.touch(tg_id)
        if message.contact.user_id and message.contact.user_id != tg_id:
            await message.answer(messages.NEW_PHOTO_REQUIRED_TEXT, reply_markup=keyboards.remove_reply_keyboard())
            return
        user = await db.get_user(tg_id)
        if user and user.get("phone"):
            await message.answer(messages.CONTACT_ALREADY_SAVED_TEXT, reply_markup=keyboards.remove_reply_keyboard())
            return
        await db.set_phone(tg_id, message.contact.phone_number)
        await db.log_event(
            tg_id,
            "contact_shared",
            {"phone": message.contact.phone_number},
        )
        await message.answer(messages.CONTACT_SHARED_TEXT, reply_markup=keyboards.remove_reply_keyboard())
