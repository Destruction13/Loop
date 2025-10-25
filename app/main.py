"""Application entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from app import messages_ru as msg
from app.config import load_settings
from app.fsm import setup_router
from app.keyboards import start_keyboard
from app.logging_conf import EVENT_ID, setup_logging
from app.services.catalog_google import GoogleCatalogConfig, GoogleSheetCatalog
from app.services.collage import CollageService
from app.services.repository import Repository
from app.services.scheduler import ReminderScheduler
from app.services.storage_local import LocalStorage
from app.services.tryon_mock import MockTryOnService
from app.services.tryon_nanobanana import NanoBananaTryOnService
from app.utils.paths import ensure_dir


async def main() -> None:
    settings = load_settings()
    logger = setup_logging()

    ensure_dir(settings.uploads_root)
    ensure_dir(settings.results_root)

    bot = Bot(token=settings.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    repository = Repository(Path("loop.db"), settings.daily_try_limit)
    await repository.init()

    storage = LocalStorage(settings.uploads_root, settings.results_root)
    catalog_config = GoogleCatalogConfig(
        csv_url=str(settings.sheet_csv_url),
        cache_ttl_seconds=settings.csv_fetch_ttl_sec,
        retries=settings.csv_fetch_retries,
    )
    catalog_service = GoogleSheetCatalog(catalog_config)

    if settings.mock_tryon:
        tryon_service = MockTryOnService(storage)
    else:
        tryon_service = NanoBananaTryOnService(
            api_url=settings.nano_api_url or "",
            api_key=settings.nano_api_key or "",
        )

    collage_service = CollageService(
        enabled=settings.collage_enabled,
        max_width=settings.collage_max_width,
        padding_px=settings.collage_padding_px,
        cache_ttl_sec=settings.collage_cache_ttl_sec,
        draw_divider=settings.collage_draw_divider,
    )

    router = setup_router(
        repository=repository,
        catalog=catalog_service,
        tryon=tryon_service,
        storage=storage,
        collage=collage_service,
        reminder_hours=settings.reminder_hours,
        selection_button_title_max=settings.button_title_max,
    )
    dp.include_router(router)

    scheduler = ReminderScheduler(
        bot=bot,
        repository=repository,
        message_text=msg.REMINDER_MESSAGE,
        keyboard_factory=start_keyboard,
        interval_seconds=60,
    )
    scheduler.start()

    logger.info("%s Bot started", EVENT_ID["START"])
    try:
        await dp.start_polling(bot)
    finally:
        await scheduler.stop()
        await asyncio.gather(catalog_service.aclose(), collage_service.aclose())
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
