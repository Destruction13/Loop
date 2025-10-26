from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from dotenv import load_dotenv

from bot.generator import GeneratorConfig, TryOnGenerator
from bot.handlers.callbacks import CallbackConfig, register_callback_handlers
from bot.handlers.contact import register_contact_handler
from bot.handlers.photo import PhotoHandlerConfig, register_photo_handlers
from bot.handlers.start import StartConfig, register_start_handlers
from bot.idle import IdleConfig, IdleWatcher
from db.init import Database
from integrations.sheets import SheetsConfig, SheetsExporter


@dataclass(slots=True)
class AppConfig:
    bot_token: str
    phone_request_after: int
    delete_old_message_on_select: bool
    idle_to_ecom_seconds: int
    idle_to_social_seconds: int
    promo_video_1: Path | None
    promo_video_2: Path | None
    catalog_root: Path
    uploads_dir: Path
    results_dir: Path
    collage: GeneratorConfig
    ecom_url: str
    cta_ecom_button_text: str
    cta_more_button_text: str
    socials_text: str
    socials_buttons: list[tuple[str, str]]
    sheets: SheetsConfig
    log_level: str


def load_config() -> AppConfig:
    load_dotenv()

    def _path(value: str | None) -> Path | None:
        if value:
            path = Path(value)
            return path if path.exists() else path
        return None

    bot_token = os.environ.get("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is required")

    phone_request_after = int(os.environ.get("PHONE_REQUEST_AFTER", "2"))
    delete_old_message = os.environ.get("DELETE_OLD_MESSAGE_ON_SELECT", "0") == "1"
    idle_to_ecom_seconds = int(float(os.environ.get("IDLE_TO_ECOM_MIN", "5")) * 60)
    idle_to_social_seconds = int(float(os.environ.get("IDLE_TO_SOCIAL_AD_MIN", "20")) * 60)

    promo_video_1 = _path(os.environ.get("PROMO_VIDEO_1"))
    promo_video_2 = _path(os.environ.get("PROMO_VIDEO_2"))

    catalog_root = Path(os.environ.get("CATALOG_ROOT", "./catalog"))
    uploads_dir = Path(os.environ.get("UPLOADS_DIR", "./uploads"))
    results_dir = Path(os.environ.get("RESULTS_DIR", "./results"))

    collage_config = GeneratorConfig(
        catalog_root=catalog_root,
        collage_width=int(os.environ.get("COLLAGE_WIDTH", "1600")),
        collage_height=int(os.environ.get("COLLAGE_HEIGHT", "800")),
        background_color=os.environ.get("COLLAGE_BACKGROUND", "#FFFFFF"),
        margin=int(os.environ.get("COLLAGE_MARGIN", "32")),
        divider_width=int(os.environ.get("COLLAGE_DIVIDER_WIDTH", "6")),
        divider_color=os.environ.get("COLLAGE_DIVIDER_COLOR", "#E5E5E5"),
    )

    ecom_url = os.environ.get("ECOM_URL", "https://example.com")
    cta_ecom_button_text = os.environ.get("CTA_ECOM_BUTTON_TEXT", "Выбрать оправу")
    cta_more_button_text = os.environ.get("CTA_TRY_MORE_BUTTON_TEXT", "Примерить ещё")

    socials_text = os.environ.get("SOCIALS_TEXT", "Мы в Instagram и TikTok — залетай!")
    socials_buttons = [
        (
            os.environ.get("SOCIALS_BUTTON_1_TEXT", "Instagram"),
            os.environ.get("SOCIALS_BUTTON_1_URL", "https://instagram.com/loov"),
        ),
        (
            os.environ.get("SOCIALS_BUTTON_2_TEXT", "TikTok"),
            os.environ.get("SOCIALS_BUTTON_2_URL", "https://tiktok.com/@loov"),
        ),
    ]

    sheets_config = SheetsConfig(
        spreadsheet_id=os.environ.get("SHEETS_SPREADSHEET_ID", ""),
        service_account_file=Path(os.environ.get("SHEETS_SERVICE_ACCOUNT_FILE", "./service_account.json")),
        interval_seconds=int(float(os.environ.get("SHEETS_EXPORT_INTERVAL_MIN", "10")) * 60),
    )

    log_level = os.environ.get("LOG_LEVEL", "INFO")

    return AppConfig(
        bot_token=bot_token,
        phone_request_after=phone_request_after,
        delete_old_message_on_select=delete_old_message,
        idle_to_ecom_seconds=idle_to_ecom_seconds,
        idle_to_social_seconds=idle_to_social_seconds,
        promo_video_1=promo_video_1,
        promo_video_2=promo_video_2,
        catalog_root=catalog_root,
        uploads_dir=uploads_dir,
        results_dir=results_dir,
        collage=collage_config,
        ecom_url=ecom_url,
        cta_ecom_button_text=cta_ecom_button_text,
        cta_more_button_text=cta_more_button_text,
        socials_text=socials_text,
        socials_buttons=socials_buttons,
        sheets=sheets_config,
        log_level=log_level,
    )


async def main() -> None:
    config = load_config()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    bot = Bot(config.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    db = Database(Path("loop.db"))
    await db.init(Path("db/schema.sql"))

    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    generator = TryOnGenerator(config.collage)
    idle_config = IdleConfig(
        ecom_timeout=config.idle_to_ecom_seconds,
        social_timeout=config.idle_to_social_seconds,
        ecom_url=config.ecom_url,
        ecom_button_text=config.cta_ecom_button_text,
        more_button_text=config.cta_more_button_text,
        socials_text=config.socials_text,
        socials_buttons=config.socials_buttons,
    )
    idle_watcher = IdleWatcher(bot, db, idle_config)

    router = Router()
    register_start_handlers(router, db, idle_watcher, StartConfig(config.promo_video_1, config.promo_video_2))
    register_photo_handlers(
        router,
        db,
        generator,
        idle_watcher,
        PhotoHandlerConfig(config.uploads_dir, config.phone_request_after),
    )
    register_callback_handlers(
        router,
        db,
        idle_watcher,
        CallbackConfig(delete_old_message_on_select=config.delete_old_message_on_select),
    )
    register_contact_handler(router, db, idle_watcher)

    dp.include_router(router)

    sheets_exporter = SheetsExporter(config.sheets)
    sheets_stop_event = asyncio.Event()
    export_task = asyncio.create_task(sheets_exporter.run(db, sheets_stop_event))

    try:
        await dp.start_polling(bot)
    finally:
        sheets_stop_event.set()
        await idle_watcher.stop()
        await export_task
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
