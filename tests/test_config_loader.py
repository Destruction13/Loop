from pathlib import Path

from app.config import load_config


def test_load_config_reads_env(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BOT_TOKEN=token123",
                "SHEET_CSV_URL=https://example.com/catalog.csv",
                "LANDING_URL=https://landing.example.com",
                'PROMO_CODE="SPECIAL"',
                "DAILY_TRY_LIMIT=9",
                "REMINDER_HOURS=12",
                "MOCK_TRYON=0",
                "CSV_FETCH_TTL_SEC=120",
                "CSV_FETCH_RETRIES=4",
                "UPLOADS_ROOT=./var/uploads",
                "RESULTS_ROOT=./var/results",
                "BUTTON_TITLE_MAX=42",
                "NANO_API_URL=https://nano.example.com",
                "NANO_API_KEY=secret",
                "COLLAGE_WIDTH=640",
                "COLLAGE_HEIGHT=320",
                "COLLAGE_GAP=12",
                "COLLAGE_PADDING=20",
                "COLLAGE_BG=#EFEFEF",
                "COLLAGE_JPEG_QUALITY=80",
                "COLLAGE_FIT_MODE=cover",
                "COLLAGE_SHARPEN=0.5",
                "COLLAGE_DIVIDER=8",
                "COLLAGE_DIVIDER_COLOR=#112233",
                "COLLAGE_DIVIDER_RADIUS=4",
                "COLLAGE_CELL_BORDER=2",
                "COLLAGE_CELL_BORDER_COLOR=#445566",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("BOT_TOKEN", raising=False)
    config = load_config(str(env_file))

    assert config.bot_token == "token123"
    assert config.sheet_csv_url == "https://example.com/catalog.csv"
    assert config.landing_url == "https://landing.example.com"
    assert config.promo_code == "SPECIAL"
    assert config.daily_try_limit == 9
    assert config.reminder_hours == 12
    assert config.mock_tryon is False
    assert config.csv_fetch_ttl_sec == 120
    assert config.csv_fetch_retries == 4
    assert config.uploads_root == Path("./var/uploads")
    assert config.results_root == Path("./var/results")
    assert config.button_title_max == 42
    assert config.nano_api_url == "https://nano.example.com"
    assert config.nano_api_key == "secret"
    assert config.collage.width == 640
    assert config.collage.height == 320
    assert config.collage.gap == 12
    assert config.collage.padding == 20
    assert config.collage.background == "#EFEFEF"
    assert config.collage.jpeg_quality == 80
    assert config.collage.fit_mode == "cover"
    assert config.collage.sharpen == 0.5
    assert config.collage.divider == 8
    assert config.collage.divider_color == "#112233"
    assert config.collage.divider_radius == 4
    assert config.collage.cell_border == 2
    assert config.collage.cell_border_color == "#445566"
