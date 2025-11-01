"""Logging configuration for the bot."""

from __future__ import annotations

import logging
import sys
from logging import Logger


EVENT_ID = {
    "START": 1000,
    "FILTER_SELECTED": 1001,
    "PHOTO_RECEIVED": 1002,
    "MODELS_SENT": 1003,
    "GENERATION_STARTED": 1004,
    "GENERATION_SUCCESS": 1005,
    "GENERATION_FAILED": 1006,
    "LIMIT_REACHED": 1007,
    "REMINDER_SCHEDULED": 1008,
    "REMINDER_SENT": 1009,
}


def setup_logging() -> Logger:
    """Configure root logging for the application."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("loop_bot")
    return logger
