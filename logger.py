from __future__ import annotations

import asyncio
import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime
from logging import Filter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

from rich.console import Console
from rich.logging import RichHandler


_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_USER_ID: ContextVar[int | str | None] = ContextVar("user_id", default=None)


class _ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges contextvars with per-call extras."""

    def process(self, msg: str, kwargs: MutableMapping[str, Any]):
        extra: Dict[str, Any] = dict(kwargs.get("extra") or {})

        module_name = self.extra.get("module_name") or self.logger.name
        extra.setdefault("module_name", module_name)

        request_id = kwargs.pop("request_id", None) or extra.pop("request_id", None)
        user_id = kwargs.pop("user_id", None) or extra.pop("user_id", None)
        stage = kwargs.pop("stage", None) or extra.pop("stage", None)
        payload = kwargs.pop("payload", None) or extra.pop("payload", None)
        suppress_sheet = kwargs.pop("suppress_sheet", None) or extra.pop("suppress_sheet", None)

        if request_id is None:
            request_id = _REQUEST_ID.get()
        if user_id is None:
            user_id = _USER_ID.get()

        if request_id is not None:
            extra.setdefault("request_id", request_id)
        if user_id is not None:
            extra.setdefault("user_id", user_id)
        if stage is not None:
            extra.setdefault("stage", stage)
        if payload is not None:
            extra.setdefault("payload", payload)
        if suppress_sheet is not None:
            extra.setdefault("suppress_sheet", suppress_sheet)

        kwargs["extra"] = extra
        return msg, kwargs


class _CompactFormatter(logging.Formatter):
    """Formatter that renders compact coloured records for RichHandler."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        time_str = self.formatTime(record, self.datefmt)
        module_name = getattr(record, "module_name", record.name)
        level = record.levelname
        message = record.message

        context_parts: list[str] = []
        request_id = getattr(record, "request_id", None)
        user_id = getattr(record, "user_id", None)
        stage = getattr(record, "stage", None)
        payload = getattr(record, "payload", None)

        if request_id:
            context_parts.append(f"rid={request_id}")
        if user_id:
            context_parts.append(f"user_id={user_id}")
        if stage:
            context_parts.append(f"stage={stage}")
        if payload:
            payload_repr = _stringify_payload(payload)
            if payload_repr:
                context_parts.append(payload_repr)

        context_suffix = f" ({', '.join(context_parts)})" if context_parts else ""

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"

        return f"[{time_str}] [{level}] [{module_name}] {message}{context_suffix}"


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        return str(payload)


class _SheetLoggingHandler(logging.Handler):
    """Handler that forwards WARNING+ records to Google Sheets."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - side effect handler
        if record.levelno < logging.WARNING:
            return
        if getattr(record, "suppress_sheet", False):
            return
        if not _SheetAppender.enabled:
            return
        message = record.getMessage()
        module = getattr(record, "module_name", record.name)
        stage = getattr(record, "stage", None)
        user_id = getattr(record, "user_id", None)
        payload = getattr(record, "payload", None)
        payload_text = _stringify_payload(payload)
        try:
            _SheetAppender.schedule_append(
                level=record.levelname,
                module=module,
                message=message,
                user_id=str(user_id) if user_id is not None else "",
                stage=str(stage) if stage else "",
                extra_text=payload_text,
            )
        except Exception:  # pragma: no cover - defensive logging
            _INTERNAL_LOGGER.exception("Не удалось поставить задачу логирования в Google Sheets")


class _SheetAppender:
    enabled = os.getenv("ENABLE_SHEET_LOGGING", "1") not in {"0", "false", "False"}
    sheet_id = os.getenv("LOG_SHEET_ID")
    sheet_name = os.getenv("LOG_SHEET_NAME", "Errors")
    service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    _worksheet: Any | None = None
    _lock: asyncio.Lock = asyncio.Lock()
    _initialization_failed = False

    @classmethod
    def schedule_append(
        cls,
        *,
        level: str,
        module: str,
        message: str,
        user_id: str,
        stage: str,
        extra_text: str,
    ) -> None:
        if not cls.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(
                cls._append(level=level, module=module, message=message, user_id=user_id, stage=stage, extra_text=extra_text)
            )
            return
        loop.create_task(
            cls._append(level=level, module=module, message=message, user_id=user_id, stage=stage, extra_text=extra_text)
        )

    @classmethod
    async def _append(
        cls,
        *,
        level: str,
        module: str,
        message: str,
        user_id: str,
        stage: str,
        extra_text: str,
    ) -> None:
        if not cls.enabled:
            return
        if not cls.sheet_id or not cls.service_account_path:
            if not cls._initialization_failed:
                _INTERNAL_LOGGER.warning(
                    "Логирование в Google Sheets отключено: не заданы LOG_SHEET_ID или GOOGLE_SERVICE_ACCOUNT_JSON",
                    extra={"suppress_sheet": True},
                )
                cls._initialization_failed = True
            return
        try:
            worksheet = await cls._get_worksheet()
        except Exception as exc:  # pragma: no cover - network/IO
            if not cls._initialization_failed:
                _INTERNAL_LOGGER.error(
                    "Не удалось подготовить таблицу для логирования: %s", exc, extra={"suppress_sheet": True}
                )
                cls._initialization_failed = True
            return

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, level, module, message, user_id, stage, extra_text]
        try:
            await asyncio.to_thread(worksheet.append_row, row, value_input_option="RAW")
        except Exception as exc:  # pragma: no cover - network/IO
            _INTERNAL_LOGGER.error(
                "Ошибка записи лога в Google Sheets: %s", exc, extra={"suppress_sheet": True}
            )

    @classmethod
    async def _get_worksheet(cls):  # pragma: no cover - network/IO
        async with cls._lock:
            if cls._worksheet is not None:
                return cls._worksheet
            import gspread

            path = Path(cls.service_account_path)
            if not path.exists():
                raise FileNotFoundError(f"Файл сервисного аккаунта не найден: {path}")

            client = await asyncio.to_thread(gspread.service_account, filename=str(path))
            spreadsheet = await asyncio.to_thread(client.open_by_key, cls.sheet_id)
            try:
                worksheet = await asyncio.to_thread(spreadsheet.worksheet, cls.sheet_name)
            except gspread.WorksheetNotFound:
                worksheet = await asyncio.to_thread(
                    spreadsheet.add_worksheet, title=cls.sheet_name, rows=10, cols=7
                )
            cls._worksheet = worksheet
            return worksheet


_INTERNAL_LOGGER = logging.getLogger("loov.logger.internal")


class _DomainInfoFilter(Filter):
    """Allow INFO records only when marked as domain milestones."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - standard interface
        if record.levelno != logging.INFO:
            return True
        return bool(getattr(record, "domain", False))


def setup_logging() -> logging.Logger:
    """Configure logging once for the entire application."""

    root = logging.getLogger()
    if getattr(root, "_loov_configured", False):
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_name)
    except AttributeError:
        log_level = logging.INFO

    root.setLevel(log_level)

    formatter_console = _CompactFormatter(datefmt="%H:%M:%S")

    console = Console()
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
    )
    console_handler.setFormatter(formatter_console)

    noise_mode = os.getenv("LOG_NOISE", "low").strip().lower() or "low"
    if noise_mode != "debug":
        console_handler.addFilter(_DomainInfoFilter())

    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_CompactFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    sheet_handler = _SheetLoggingHandler()
    sheet_handler.setLevel(logging.WARNING)

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.addHandler(sheet_handler)

    # Silence verbose third-party loggers.
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("aiogram.dispatcher").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    root._loov_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.LoggerAdapter:
    base = logging.getLogger(name)
    return _ContextLoggerAdapter(base, {"module_name": name})


def info_domain(
    module: str,
    message: str,
    *,
    stage: str | None = None,
    user_id: int | str | None = None,
    **context: Any,
) -> None:
    """Log a milestone INFO message visible in console output."""

    logger = get_logger(module)
    extra: Dict[str, Any] = {"domain": True}
    if stage:
        extra["stage"] = stage
    if context:
        extra["payload"] = context
    kwargs: Dict[str, Any] = {"extra": extra}
    if user_id is not None:
        kwargs["user_id"] = user_id
    logger.info(message, **kwargs)


def log_event(
    level: str | int,
    module: str,
    message: str,
    *,
    user_id: int | str | None = None,
    stage: str | None = None,
    extra: Mapping[str, Any] | None = None,
    exc_info: Any | None = None,
) -> None:
    logger = get_logger(module)
    kwargs: Dict[str, Any] = {}
    if extra:
        kwargs["extra"] = {"payload": dict(extra)}
    if user_id is not None:
        kwargs["user_id"] = user_id
    if stage:
        kwargs["stage"] = stage
    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = int(level)
    logger.log(level_value, message, exc_info=exc_info, **kwargs)


def bind_context(*, request_id: str | None = None, user_id: int | str | None = None) -> Dict[str, Any]:
    tokens: Dict[str, Any] = {}
    if request_id is not None:
        tokens["request_id"] = _REQUEST_ID.set(request_id)
    if user_id is not None:
        tokens["user_id"] = _USER_ID.set(user_id)
    return tokens


def reset_context(tokens: Mapping[str, Any]) -> None:
    request_token = tokens.get("request_id")
    if request_token is not None:
        _REQUEST_ID.reset(request_token)
    user_token = tokens.get("user_id")
    if user_token is not None:
        _USER_ID.reset(user_token)


__all__ = [
    "setup_logging",
    "get_logger",
    "info_domain",
    "log_event",
    "bind_context",
    "reset_context",
]
