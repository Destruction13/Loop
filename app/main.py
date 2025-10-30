"""Application entry point."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from app.config import load_config
from app.fsm import setup_router
from app.keyboards import reminder_keyboard, start_keyboard
from app.logging_conf import EVENT_ID, setup_logging
from app.services import nanobanana
from app.services.catalog_google import GoogleCatalogConfig, GoogleSheetCatalog
from app.services.collage import build_three_tile_collage
from app.services.contact_export import ContactSheetExporter
from app.services.leads_export import LeadsExporter
from app.services.repository import Repository
from app.services.social_ad import SocialAdService
from app.services.scheduler import ReminderScheduler
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
    tmp_dir = ensure_dir(Path("tmp"))
    ensure_dir(Path(".cache") / "frames")

    async def _cleanup_tmp() -> None:
        def _clean() -> None:
            cutoff = time.time() - 3600
            if not tmp_dir.exists():
                return
            for entry in tmp_dir.iterdir():
                try:
                    if entry.is_file() and entry.stat().st_mtime < cutoff:
                        entry.unlink(missing_ok=True)
                except OSError:
                    continue

        await asyncio.to_thread(_clean)

    asyncio.create_task(_cleanup_tmp())

    nanobanana.configure(config.nanobanana_api_key)

    bot = Bot(token=config.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    repository = Repository(Path("loop.db"), config.daily_try_limit)
    await repository.init()

    catalog_config = GoogleCatalogConfig(
        csv_url=str(config.sheet_csv_url),
        cache_ttl_seconds=config.csv_fetch_ttl_sec,
        retries=config.csv_fetch_retries,
        parse_row_limit=config.catalog_row_limit,
    )
    catalog_service = GoogleSheetCatalog(catalog_config)

    recommendation_settings = RecommendationSettings(
        batch_size=config.batch_size,
        pick_scheme=PickScheme.from_string(config.pick_scheme),
        clear_on_catalog_change=config.reco_clear_on_catalog_change,
    )
    recommender = RecommendationService(
        catalog=catalog_service,
        repository=repository,
        settings=recommendation_settings,
    )

    leads_exporter = LeadsExporter(
        enabled=config.enable_leads_export,
        sheet_name=config.leads_sheet_name,
        promo_code=config.promo_contact_code,
    )

    contact_exporter = ContactSheetExporter(
        sheet_url=config.contacts_sheet_url or "",
        worksheet_name="Контакты",
        credentials_path=config.google_service_account_json,
    )

    router = setup_router(
        repository=repository,
        recommender=recommender,
        collage_config=config.collage,
        collage_builder=build_three_tile_collage,
        batch_size=config.batch_size,
        reminder_hours=config.reminder_hours,
        selection_button_title_max=config.button_title_max,
        site_url=str(config.site_url),
        promo_code=config.promo_code,
        no_more_message_key=config.reco_no_more_key,
        contact_reward_rub=config.contact_reward_rub,
        promo_contact_code=config.promo_contact_code,
        leads_exporter=leads_exporter,
        contact_exporter=contact_exporter,
        idle_nudge_seconds=max(config.idle_reminder_minutes, 0) * 60,
        enable_idle_nudge=config.enable_idle_reminder,
        privacy_policy_url=str(config.privacy_policy_url),
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

    social_ad: SocialAdService | None = None
    if config.enable_social_ad and config.social_ad_minutes > 0:
        social_ad = SocialAdService(
            bot=bot,
            repository=repository,
            instagram_url=str(config.social_instagram_url),
            tiktok_url=str(config.social_tiktok_url),
            timeout_minutes=config.social_ad_minutes,
            interval_seconds=30,
        )
        social_ad.start()

    logger.info("%s Bot started", EVENT_ID["START"])
    try:
        await dp.start_polling(bot)
    finally:
        await scheduler.stop()
        if social_ad is not None:
            await social_ad.stop()
        await catalog_service.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
