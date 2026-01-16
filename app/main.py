"""Application entry point."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from contextlib import suppress
from logging import Logger
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, MenuButtonCommands
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from app.analytics import (
    AnalyticsExporter,
    AnalyticsExporterConfig,
    init as analytics_init,
)
from app.config import load_config
from app.fsm import setup_router
from app.keyboards import event_trigger_keyboard, reminder_keyboard
from app.services import nanobanana
from app.services.catalog_google import GoogleCatalogConfig, GoogleSheetCatalog
from app.services.collage import build_three_tile_collage
from app.services.contact_export import ContactSheetExporter
from app.services.event_teaser import (
    EVENT_TEASER_ALREADY_SENT,
    EVENT_TEASER_FAILED,
    EVENT_TEASER_RATE_LIMITED,
    EVENT_TEASER_SENT,
    maybe_send_event_teaser,
)
from app.services.leads_export import LeadsExporter
from app.services.repository import Repository
from app.services.social_ad import SocialAdService
from app.services.scheduler import ReminderScheduler
from app.texts import messages as msg
from app.utils.paths import ensure_dir
from app.recommender import StyleRecommender
from app.infrastructure.logging_middleware import LoggingMiddleware
from app.infrastructure.identity_middleware import UserIdentityMiddleware
from app.admin.api import start_admin_api, stop_admin_api
from app.infrastructure.tunnel import (
    CloudflareTunnel,
    start_tunnel_and_update_config,
)
from logger import get_logger, info_domain, log_event, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def _run_polling(dp: Dispatcher, bot: Bot, logger: Logger) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    stop_waiter = asyncio.create_task(stop_event.wait())

    def _handle_signal(sig: signal.Signals) -> None:
        logger.debug("Received %s signal. Shutting down...", sig.name)
        stop_event.set()

    signals = (signal.SIGINT, signal.SIGTERM)
    for sig in signals:
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - platform specific
            continue

    polling_task = asyncio.create_task(
        dp.start_polling(bot, handle_signals=False), name="aiogram-polling"
    )
    polling_task.add_done_callback(lambda _: stop_event.set())

    done, _pending = await asyncio.wait(
        {polling_task, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
    )

    if stop_waiter in done and not polling_task.done():
        logger.debug("Stopping polling loop gracefully...")
        await dp.stop_polling()

    await asyncio.gather(polling_task, return_exceptions=False)

    stop_waiter.cancel()
    with suppress(asyncio.CancelledError):
        await stop_waiter

    for sig in signals:
        try:
            loop.remove_signal_handler(sig)
        except (ValueError, RuntimeError):  # pragma: no cover - platform specific
            continue


async def _broadcast_event_teaser(
    *,
    bot: Bot,
    repository: Repository,
    logger: Logger,
    event_id: str,
    image_path: Path,
) -> None:
    if not event_id:
        return
    already_done = await repository.has_event_teaser_broadcast(event_id)
    if already_done:
        info_domain(
            "event",
            "Event teaser broadcast already completed",
            stage="EVENT_TRIGGER_BROADCAST_SKIPPED",
            event_id=event_id,
        )
        return
    user_ids = await repository.list_event_teaser_candidates()
    total = len(user_ids)
    if total <= 0:
        await repository.mark_event_teaser_broadcast(event_id)
        info_domain(
            "event",
            "Event teaser broadcast finished",
            stage="EVENT_TRIGGER_BROADCAST_DONE",
            event_id=event_id,
            total=0,
            sent=0,
            skipped=0,
            errors=0,
        )
        return
    if not image_path.exists():
        logger.warning("Event teaser image missing: %s", image_path)
    info_domain(
        "event",
        "Event teaser broadcast started",
        stage="EVENT_TRIGGER_BROADCAST_START",
        event_id=event_id,
        total=total,
    )
    sent = 0
    skipped = 0
    errors = 0
    keyboard = event_trigger_keyboard()
    for user_id in user_ids:
        attempts = 0
        while True:
            attempts += 1
            try:
                result = await maybe_send_event_teaser(
                    bot=bot,
                    repository=repository,
                    user_id=user_id,
                    event_id=event_id,
                    text=msg.EVENT_TRIGGER_TEXT,
                    reply_markup=keyboard,
                    image_path=image_path,
                    send_teaser=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.warning(
                    "Event teaser broadcast failed for %s: %s", user_id, exc
                )
                break
            if result.status == EVENT_TEASER_RATE_LIMITED and result.retry_after:
                if attempts >= 3:
                    errors += 1
                    break
                await asyncio.sleep(result.retry_after)
                continue
            if result.status == EVENT_TEASER_SENT:
                sent += 1
            elif result.status == EVENT_TEASER_ALREADY_SENT:
                skipped += 1
            elif result.status == EVENT_TEASER_FAILED:
                errors += 1
            break
    await repository.mark_event_teaser_broadcast(event_id)
    info_domain(
        "event",
        "Event teaser broadcast finished",
        stage="EVENT_TRIGGER_BROADCAST_DONE",
        event_id=event_id,
        total=total,
        sent=sent,
        skipped=skipped,
        errors=errors,
    )


async def main() -> None:
    config = load_config()
    setup_logging()
    logger = get_logger("bot.start")

    info_domain(
        "bot.start",
        "Конфиг загружен",
        stage="CONFIG_OK",
        leads_export=config.enable_leads_export,
        idle_reminder=config.enable_idle_reminder,
        social_ad=config.enable_social_ad,
    )

    ensure_dir((PROJECT_ROOT / config.uploads_root).resolve())
    ensure_dir((PROJECT_ROOT / config.results_root).resolve())
    tmp_dir = ensure_dir(PROJECT_ROOT / "var" / "tmp")
    ensure_dir(PROJECT_ROOT / ".cache" / "frames")

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

    nanobanana.configure(config.nanobanana_key_slots)

    bot = Bot(
    token=config.bot_token,
    default=DefaultBotProperties(
        parse_mode=ParseMode.HTML
    )
)
    commands = [
        BotCommand(command="start", description="🚀 Запуск бота"),
        BotCommand(command="wear", description="👓 Примерить новые очки"),
        BotCommand(command="help", description="❓ Как это работает"),
        BotCommand(command="privacy", description="🔒 Политика конфиденциальности"),
    ]
    if config.event_enabled:
        commands.append(BotCommand(command="event", description="🎄 Ивент"))
        commands.insert(2, commands.pop())
    await bot.set_my_commands(commands)
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    repository_path = (PROJECT_ROOT / "loop.db").resolve()
    repository = Repository(repository_path, config.daily_try_limit)
    await repository.init()
    event_ready = bool(
        config.event_enabled
        and config.event_id
        and config.event_scenes_sheet
        and config.event_prompt_json
    )
    if config.event_enabled and event_ready:
        await _broadcast_event_teaser(
            bot=bot,
            repository=repository,
            logger=logger,
            event_id=config.event_id,
            image_path=PROJECT_ROOT / "images" / "event_new_year.jpg",
        )

    dp = Dispatcher()
    dp.update.middleware(LoggingMiddleware())
    dp.update.middleware(UserIdentityMiddleware(repository))

    await analytics_init(repository_path)

    google_credentials_path = None
    if config.google_service_account_json is not None:
        if config.google_service_account_json.is_absolute():
            google_credentials_path = config.google_service_account_json
        else:
            google_credentials_path = (PROJECT_ROOT / config.google_service_account_json).resolve()

    promo_video_path = (
        config.promo_video_path
        if config.promo_video_path.is_absolute()
        else (PROJECT_ROOT / config.promo_video_path).resolve()
    )
    if not promo_video_path.exists():
        logger.warning(
            "Файл промо-видео не найден по пути %s",
            promo_video_path,
            extra={"stage": "PROMO_VIDEO_MISSING"},
        )

    catalog_config = GoogleCatalogConfig(
        csv_url=config.sheet_csv_url,
        sheet_id=config.catalog_sheet_id,
        sheet_gid=config.catalog_sheet_gid,
        cache_ttl_seconds=config.csv_fetch_ttl_sec,
        retries=config.csv_fetch_retries,
        parse_row_limit=config.catalog_row_limit,
    )
    catalog_service = GoogleSheetCatalog(catalog_config)

    style_recommender = StyleRecommender(repository)

    leads_exporter = LeadsExporter(
        enabled=config.enable_leads_export,
        sheet_name=config.leads_sheet_name,
        promo_code=config.promo_contact_code,
        spreadsheet_id=config.catalog_sheet_id,
        spreadsheet_url=config.contacts_sheet_url,
        credentials_path=google_credentials_path,
    )

    contact_exporter = ContactSheetExporter(
        sheet_url=config.contacts_sheet_url or "",
        worksheet_name="Контакты",
        credentials_path=google_credentials_path,
    )

    router = setup_router(
        repository=repository,
        catalog=catalog_service,
        style_recommender=style_recommender,
        collage_config=config.collage,
        collage_builder=build_three_tile_collage,
        batch_size=config.batch_size,
        reminder_hours=config.reminder_hours,
        selection_button_title_max=config.button_title_max,
        show_model_style_tag=config.show_model_style_tag,
        site_url=str(config.site_url),
        admin_webapp_url=config.admin_webapp_url,
        promo_code=config.promo_code,
        no_more_message_key=config.reco_no_more_key,
        clear_on_catalog_change=config.reco_clear_on_catalog_change,
        contact_reward_rub=config.contact_reward_rub,
        promo_contact_code=config.promo_contact_code,
        leads_exporter=leads_exporter,
        contact_exporter=contact_exporter,
        idle_nudge_seconds=max(config.idle_reminder_minutes, 0) * 60,
        enable_idle_nudge=config.enable_idle_reminder,
        privacy_policy_url=str(config.privacy_policy_url),
        promo_video_path=promo_video_path,
        promo_video_enabled=config.promo_video_enabled,
        promo_video_width=config.promo_video_width,
        promo_video_height=config.promo_video_height,
        event_enabled=config.event_enabled,
        event_id=config.event_id,
        event_scenes_sheet=config.event_scenes_sheet,
        event_prompt_json=config.event_prompt_json,
        event_debug_bundle=config.event_debug_bundle,
        event_model_name=config.event_model_name,
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

    analytics_exporter: AnalyticsExporter | None = None
    if (
        config.analytics_spreadsheet_id
        and google_credentials_path is not None
    ):
        analytics_config = AnalyticsExporterConfig(
            spreadsheet_id=config.analytics_spreadsheet_id,
            credentials_path=google_credentials_path,
            events_sheet_name=config.analytics_events_sheet_name,
            analytics_sheet_name=config.analytics_sheet_name,
            flush_interval=config.analytics_flush_interval_sec,
        )
        analytics_exporter = AnalyticsExporter(analytics_config, repository_path)
        await analytics_exporter.start()
    else:
        logger.warning(
            "Событийная аналитика не запущена: нет access_id или ключей",
            extra={"stage": "ANALYTICS_DISABLED"},
        )

    social_ad: SocialAdService | None = None
    if config.enable_social_ad and config.social_ad_minutes > 0:
        social_ad = SocialAdService(
            bot=bot,
            repository=repository,
            social_links=[(link.title, link.url) for link in config.social_links],
            timeout_minutes=config.social_ad_minutes,
            interval_seconds=30,
        )
        social_ad.start()

    # Start embedded Admin API server
    admin_api_runner = None
    admin_api_port = int(os.getenv("ADMIN_API_PORT", "8080"))
    try:
        admin_api_runner, _ = await start_admin_api(
            config=config,
            db_path=repository_path,
        )
        info_domain(
            "bot.start",
            f"Admin API запущен на порту {admin_api_port}",
            stage="ADMIN_API_STARTED",
            port=admin_api_port,
        )
    except Exception as exc:
        logger.warning(
            "Не удалось запустить Admin API: %s",
            exc,
            extra={"stage": "ADMIN_API_FAILED"},
        )

    # Start Cloudflare Tunnel for HTTPS access to Admin API
    tunnel: CloudflareTunnel | None = None
    tunnel_url: str | None = None
    admin_webapp_url_override: str | None = None
    if admin_api_runner is not None:
        if CloudflareTunnel.is_available():
            info_domain(
                "bot.start",
                "Запуск Cloudflare Tunnel...",
                stage="TUNNEL_STARTING",
            )
            tunnel, tunnel_url = await start_tunnel_and_update_config(
                local_port=admin_api_port,
                timeout=30.0,
            )
            if tunnel_url:
                info_domain(
                    "bot.start",
                    f"Cloudflare Tunnel запущен: {tunnel_url}",
                    stage="TUNNEL_STARTED",
                    tunnel_url=tunnel_url,
                )
                # Use tunnel URL for admin panel (serves both frontend and API)
                admin_webapp_url_override = f"{tunnel_url}/admin"
                info_domain(
                    "bot.start",
                    f"🔐 Admin панель доступна: {admin_webapp_url_override}",
                    stage="ADMIN_PANEL_URL",
                    admin_url=admin_webapp_url_override,
                )
                # Update the router with the new admin URL
                router.admin_webapp_url = admin_webapp_url_override
            else:
                logger.warning(
                    "Не удалось запустить Cloudflare Tunnel. "
                    "Admin API доступен только локально на порту %s",
                    admin_api_port,
                    extra={"stage": "TUNNEL_FAILED"},
                )
        else:
            logger.warning(
                "cloudflared не установлен. Admin API доступен только локально. "
                "Установите cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
                extra={"stage": "CLOUDFLARED_NOT_INSTALLED"},
            )

    info_domain("bot.start", "Бот запущен", stage="BOT_STARTED")
    try:
        await _run_polling(dp, bot, logger)
    finally:
        await scheduler.stop()
        if social_ad is not None:
            await social_ad.stop()
        if analytics_exporter is not None:
            await analytics_exporter.stop()
        if tunnel is not None:
            await tunnel.stop()
        if admin_api_runner is not None:
            await stop_admin_api(admin_api_runner)
        await catalog_service.aclose()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        setup_logging()
        log_event(
            "CRITICAL",
            "bot.runtime",
            f"Необработанное исключение: {exc}",
            stage="UNHANDLED_EXCEPTION",
            extra={"exception": repr(exc)},
        )
        raise
