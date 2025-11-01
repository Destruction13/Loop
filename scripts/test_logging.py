from __future__ import annotations

import asyncio

from logger import get_logger, log_event, setup_logging


async def main() -> None:
    setup_logging()
    logger = get_logger("scripts.test")
    logger.info("🧪 Тестовый информационный лог", extra={"stage": "TEST_INFO"})
    log_event(
        "WARNING",
        "scripts.test",
        "⚠️ Тестовое предупреждение",
        stage="TEST_WARNING",
        extra={"details": "Проверка записи в Google Sheets"},
    )
    log_event(
        "ERROR",
        "scripts.test",
        "❌ Тестовая ошибка",
        stage="TEST_ERROR",
        extra={"details": "Проверка обработки ошибок"},
    )
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
