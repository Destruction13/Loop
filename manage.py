#!/usr/bin/env python3
"""Utility CLI for managing the LOOV Telegram bot."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{10,}$")
CSV_CONTENT_TYPE = "text/csv"


@dataclass(slots=True)
class CheckResult:
    """Single diagnostic result entry."""

    title: str
    message: str
    status: str  # ok | warn | fail

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "fail": "❌"}.get(self.status, "❓")

    def colorize(self, text: str) -> str:
        colors = {"ok": "\033[32m", "warn": "\033[33m", "fail": "\033[31m"}
        prefix = colors.get(self.status, "")
        suffix = "\033[0m" if prefix else ""
        return f"{prefix}{text}{suffix}"

    def formatted(self) -> str:
        return self.colorize(f"{self.icon} {self.title}: {self.message}")


def _load_env() -> None:
    """Load .env values without overriding existing environment variables."""

    load_dotenv(ENV_FILE, override=False)


def _validate_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _command_run() -> int:
    from app.main import main as app_main  # Local import to avoid heavy deps for other cmds

    asyncio.run(app_main())
    return 0


def _command_check() -> int:
    _load_env()

    results: List[CheckResult] = []
    env_values: dict[str, str] = {}

    def add_result(title: str, status: str, message: str) -> None:
        results.append(CheckResult(title=title, status=status, message=message))

    # Environment presence and basic validation
    required_vars = {
        "BOT_TOKEN": "Токен Telegram-бота",
        "SHEET_CSV_URL": "Ссылка на CSV каталога",
        "LANDING_URL": "Ссылка на лендинг",
        "SOCIAL_LINKS_JSON": "JSON соцсетей",
        "NANOBANANA_API_KEY": "Ключ NanoBanana",
        "PROMO_CODE": "Промокод",
        "DAILY_TRY_LIMIT": "Лимит примерок",
        "CATALOG_ROW_LIMIT": "Лимит строк каталога",
        "PICK_SCHEME": "Режим подбора",
        "GOOGLE_SHEET_URL": "Google Sheet для лидов",
        "GOOGLE_SERVICE_ACCOUNT_JSON": "Путь к сервисному аккаунту",
    }

    for name, description in required_vars.items():
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            add_result(name, "fail", f"{description}: значение не задано")
            continue
        value = raw.strip().strip('"').strip("'")
        status = "ok"
        message = f"{description}: OK"
        if name == "BOT_TOKEN":
            if TOKEN_RE.match(value):
                env_values[name] = value
            else:
                status = "fail"
                message = "Токен выглядит некорректно"
        elif name in {"SHEET_CSV_URL", "LANDING_URL", "GOOGLE_SHEET_URL"}:
            if _validate_url(value):
                env_values[name] = value
            else:
                status = "fail"
                message = "Невалидный URL"
        elif name == "SOCIAL_LINKS_JSON":
            env_values[name] = value
            message = "JSON указан, проверим ниже"
        elif name == "DAILY_TRY_LIMIT":
            try:
                limit = int(value)
                if limit < 1:
                    raise ValueError
            except ValueError:
                status = "fail"
                message = "Должно быть целое число ≥ 1"
            else:
                env_values[name] = str(limit)
                message = f"Лимит: {limit}"
        elif name == "CATALOG_ROW_LIMIT":
            try:
                limit = int(value)
                if limit < 0:
                    raise ValueError
            except ValueError:
                status = "fail"
                message = "Должно быть целое число ≥ 0"
            else:
                env_values[name] = str(limit)
                message = f"Лимит строк: {limit}" if limit else "Лимит строк: без ограничения"
        elif name in {"PROMO_CODE", "PICK_SCHEME", "NANOBANANA_API_KEY"}:
            env_values[name] = value
        elif name == "GOOGLE_SERVICE_ACCOUNT_JSON":
            env_values[name] = value
        else:
            env_values[name] = value
        add_result(name, status, message)

    # SOCIAL_LINKS_JSON parsing
    social_raw = env_values.get("SOCIAL_LINKS_JSON")
    if social_raw is None:
        add_result("SOCIAL_LINKS_JSON format", "fail", "Переменная отсутствует")
    else:
        try:
            data = json.loads(social_raw)
            if not isinstance(data, list):
                raise ValueError("Должен быть массив объектов")
            for entry in data:
                if not isinstance(entry, dict):
                    raise ValueError("Каждый элемент должен быть объектом")
                title = str(entry.get("title") or "").strip()
                url = str(entry.get("url") or "").strip()
                if not title or not url:
                    raise ValueError("Нужны поля title и url")
                if not _validate_url(url):
                    raise ValueError(f"URL '{url}' некорректен")
        except ValueError as exc:
            add_result("SOCIAL_LINKS_JSON format", "fail", f"Ошибка разбора: {exc}")
        except json.JSONDecodeError as exc:
            add_result("SOCIAL_LINKS_JSON format", "fail", f"JSON невалиден: {exc}")
        else:
            add_result("SOCIAL_LINKS_JSON format", "ok", "JSON корректен")

    # GOOGLE_SERVICE_ACCOUNT_JSON path resolution
    service_path_result: Optional[CheckResult] = None
    creds_raw = env_values.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_raw:
        raw_path = Path(creds_raw)
        if raw_path.is_absolute():
            resolved_path = raw_path
        else:
            resolved_path = (PROJECT_ROOT / raw_path).resolve()
        if resolved_path.exists():
            service_path_result = CheckResult(
                title="Google service account",
                status="ok",
                message=f"Файл найден: {resolved_path}",
            )
        else:
            service_path_result = CheckResult(
                title="Google service account",
                status="fail",
                message=f"Файл не найден по пути: {resolved_path}",
            )
    else:
        service_path_result = CheckResult(
            title="Google service account",
            status="fail",
            message="Путь не задан",
        )
    results.append(service_path_result)

    # CSV availability check
    csv_url = env_values.get("SHEET_CSV_URL")
    if csv_url:
        try:
            response = requests.get(csv_url, timeout=5)
        except requests.RequestException as exc:
            results.append(
                CheckResult(
                    title="Каталог CSV",
                    status="warn",
                    message=f"Не удалось подключиться: {exc}",
                )
            )
        else:
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status_code == 200 and CSV_CONTENT_TYPE in content_type:
                results.append(
                    CheckResult(
                        title="Каталог CSV",
                        status="ok",
                        message="Ответ 200 и тип text/csv",
                    )
                )
            elif response.status_code == 200:
                results.append(
                    CheckResult(
                        title="Каталог CSV",
                        status="warn",
                        message=f"Ответ 200, но тип {content_type or 'не указан'}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        title="Каталог CSV",
                        status="fail",
                        message=f"HTTP {response.status_code}",
                    )
                )

    # Promo video presence
    promo_video = (PROJECT_ROOT / "video" / "promo_start.mp4").resolve()
    if promo_video.exists():
        results.append(
            CheckResult(
                title="Promo video",
                status="ok",
                message=f"Файл найден: {promo_video}",
            )
        )
    else:
        results.append(
            CheckResult(
                title="Promo video",
                status="warn",
                message=f"Файл отсутствует: {promo_video}",
            )
        )

    # Telegram connectivity
    bot_token = env_values.get("BOT_TOKEN")
    if bot_token:
        telegram_url = f"https://api.telegram.org/bot{bot_token}/getMe"
        try:
            response = requests.get(telegram_url, timeout=5)
        except requests.RequestException as exc:
            results.append(
                CheckResult(
                    title="Telegram API",
                    status="warn",
                    message=f"Не удалось проверить: {exc}",
                )
            )
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:  # pragma: no cover - неожиданный ответ
                    results.append(
                        CheckResult(
                            title="Telegram API",
                            status="warn",
                            message=f"Неожиданный формат ответа: {exc}",
                        )
                    )
                else:
                    if payload.get("ok"):
                        results.append(
                            CheckResult(
                                title="Telegram API",
                                status="ok",
                                message="getMe успешно выполнен",
                            )
                        )
                    else:
                        results.append(
                            CheckResult(
                                title="Telegram API",
                                status="warn",
                                message=f"Ответ 200, но ok={payload.get('ok')}",
                            )
                        )
            elif response.status_code == 401:
                results.append(
                    CheckResult(
                        title="Telegram API",
                        status="fail",
                        message="Токен отклонён (HTTP 401)",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        title="Telegram API",
                        status="warn",
                        message=f"Ответ {response.status_code}: {response.text[:120]}",
                    )
                )

    # Writable var/tmp directory
    tmp_dir = PROJECT_ROOT / "var" / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        probe = tmp_dir / ".selftest"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        results.append(
            CheckResult(
                title="var/tmp",
                status="fail",
                message=f"Нет доступа к записи: {exc}",
            )
        )
    else:
        results.append(
            CheckResult(
                title="var/tmp",
                status="ok",
                message=f"Каталог доступен: {tmp_dir}",
            )
        )

    # Output report
    print("\n=== Self-check отчёт ===")
    for item in results:
        print(item.formatted())

    has_fail = any(item.status == "fail" for item in results)
    has_warn = any(item.status == "warn" for item in results)

    if has_fail:
        summary = CheckResult(
            title="Итог",
            status="fail",
            message="обнаружены критические ошибки",
        )
    elif has_warn:
        summary = CheckResult(
            title="Итог",
            status="warn",
            message="есть предупреждения, но критических ошибок нет",
        )
    else:
        summary = CheckResult(
            title="Итог",
            status="ok",
            message="всё готово к запуску",
        )
    print(summary.formatted())

    return 1 if has_fail else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LOOV management CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Запустить бота")
    run_parser.set_defaults(func=lambda _args: _command_run())

    check_parser = subparsers.add_parser("check", help="Выполнить самопроверку окружения")
    check_parser.set_defaults(func=lambda _args: _command_check())

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
