"""Application configuration using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    bot_token: str = Field(..., env="BOT_TOKEN")
    daily_try_limit: int = Field(7, env="DAILY_TRY_LIMIT")
    reminder_hours: int = Field(24, env="REMINDER_HOURS")
    mock_tryon: bool = Field(True, env="MOCK_TRYON")
    catalog_root: Path = Field(Path("./catalog"), env="CATALOG_ROOT")
    uploads_root: Path = Field(Path("./uploads"), env="UPLOADS_ROOT")
    results_root: Path = Field(Path("./results"), env="RESULTS_ROOT")
    nano_api_url: Optional[str] = Field(None, env="NANO_API_URL")
    nano_api_key: Optional[str] = Field(None, env="NANO_API_KEY")
    drive_public_base_url: Optional[str] = Field(None, env="DRIVE_PUBLIC_BASE_URL")

    @validator("catalog_root", "uploads_root", "results_root", pre=True)
    def _ensure_path(cls, value: str | Path) -> Path:  # type: ignore[override]
        """Convert string values to Path objects."""

        return Path(value)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings from environment with optional .env override."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    return Settings()
