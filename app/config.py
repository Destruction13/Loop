"""Application configuration using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRT2CXRcmWxmWHKADYfHTadlxBUZ-"
    "R7nEX7HcAqrBo_PzSKYrCln4HFeCUJTB2q_C7asfwO7AOLNiwh/pub?output=csv"
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    bot_token: str = Field(..., env="BOT_TOKEN")
    sheet_csv_url: AnyHttpUrl = Field(DEFAULT_SHEET_URL, env="SHEET_CSV_URL")
    daily_try_limit: int = Field(7, env="DAILY_TRY_LIMIT")
    reminder_hours: int = Field(24, env="REMINDER_HOURS")
    mock_tryon: bool = Field(True, env="MOCK_TRYON")
    landing_url: AnyHttpUrl = Field("https://example.com/booking", env="LANDING_URL")
    promo_template: str = Field("-promocode-", env="PROMO_TEMPLATE")
    uploads_root: Path = Field(Path("./uploads"), env="UPLOADS_ROOT")
    results_root: Path = Field(Path("./results"), env="RESULTS_ROOT")
    csv_fetch_ttl_sec: int = Field(60, env="CSV_FETCH_TTL_SEC")
    csv_fetch_retries: int = Field(3, env="CSV_FETCH_RETRIES")
    collage_enabled: bool = Field(True, env="COLLAGE_ENABLED")
    collage_max_width: int = Field(1280, env="COLLAGE_MAX_WIDTH")
    collage_padding_px: int = Field(20, env="COLLAGE_PADDING_PX")
    collage_cache_ttl_sec: int = Field(300, env="COLLAGE_CACHE_TTL_SEC")
    collage_draw_divider: bool = Field(True, env="COLLAGE_DRAW_DIVIDER")
    button_title_max: int = Field(28, env="BUTTON_TITLE_MAX")
    nano_api_url: Optional[str] = Field(None, env="NANO_API_URL")
    nano_api_key: Optional[str] = Field(None, env="NANO_API_KEY")
    drive_public_base_url: Optional[str] = Field(None, env="DRIVE_PUBLIC_BASE_URL")

    @field_validator("uploads_root", "results_root", mode="before")
    def _ensure_path(cls, value: str | Path) -> Path:  # type: ignore[override]
        """Convert string values to Path objects."""

        return Path(value)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings from environment with optional .env override."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    return Settings()
