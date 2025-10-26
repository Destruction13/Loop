"""Application entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from app.config import load_config
from app.fsm import setup_router
from app.keyboards import reminder_keyboard, start_keyboard
from app.logging_conf import EVENT_ID, setup_logging
from app.services.catalog_google import GoogleCatalogConfig, GoogleSheetCatalog
from app.services.collage import build_three_tile_collage
from app.services.repository import Repository
from app.services.scheduler import ReminderScheduler
from app.services.storage_local import LocalStorage
from app.services.tryon_mock import MockTryOnService
from app.services.tryon_nanobanana import NanoBananaTryOnService
from app.texts import messages as msg
from app.utils.paths import ensure_dir
from app.services.recommendation import (
    PickScheme,
    RecommendationService,
    RecommendationSettings,
)


async def main() -> None:
    config = load_config()
    logger = setup_logging()

    ensure_dir(config.uploads_root)
    ensure_dir(config.results_root)

    bot = Bot(token=config.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    repository = Repository(Path("loop.db"), config.daily_try_limit)
    await repository.init()

    storage = LocalStorage(config.uploads_root, config.results_root)
    catalog_config = GoogleCatalogConfig(
        csv_url=str(config.sheet_csv_url),
        cache_ttl_seconds=config.csv_fetch_ttl_sec,
        retries=config.csv_fetch_retries,
    )
    catalog_service = GoogleSheetCatalog(catalog_config)

    recommendation_settings = RecommendationSettings(
        batch_size=config.batch_size,
        pick_scheme=PickScheme.from_string(config.pick_scheme),
        unique_context=config.reco_unique_context,
        clear_on_catalog_change=config.reco_clear_on_catalog_change,
    )
    recommender = RecommendationService(
        catalog=catalog_service,
        repository=repository,
        settings=recommendation_settings,
    )

    if config.mock_tryon:
        tryon_service = MockTryOnService(storage)
    else:
        tryon_service = NanoBananaTryOnService(
            api_url=config.nano_api_url or "",
            api_key=config.nano_api_key or "",
        )

    router = setup_router(
        repository=repository,
        recommender=recommender,
        tryon=tryon_service,
        storage=storage,
        collage_config=config.collage,
        collage_builder=build_three_tile_collage,
        batch_size=config.batch_size,
        reminder_hours=config.reminder_hours,
        selection_button_title_max=config.button_title_max,
        landing_url=str(config.landing_url),
        promo_code=config.promo_code,
        no_more_message_key=config.reco_no_more_key,
    )
    dp.include_router(router)

    scheduler = ReminderScheduler(
        bot=bot,
        repository=repository,
        message_text=msg.REMINDER_MESSAGE,
        keyboard_factory=reminder_keyboard,
        interval_seconds=60,
    )
    scheduler.start()

    logger.info("%s Bot started", EVENT_ID["START"])
    try:
        await dp.start_polling(bot)
    finally:
        await scheduler.stop()
        await catalog_service.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
