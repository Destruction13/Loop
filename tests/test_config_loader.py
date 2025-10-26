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
                "BATCH_SIZE=5",
                "BATCH_LAYOUT_COLS=3",
                "PICK_SCHEME=GENDER_OR_GENDER_UNISEX",
                "CANVAS_WIDTH=1500",
                "CANVAS_HEIGHT=500",
                "CANVAS_BG=#EFEFEF",
                "TILE_MARGIN=40",
                "DIVIDER_WIDTH=6",
                "DIVIDER_COLOR=#123456",
                "JPEG_QUALITY=82",
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
    assert config.batch_size == 5
    assert config.batch_layout_cols == 3
    assert config.pick_scheme == "GENDER_OR_GENDER_UNISEX"
    assert config.collage.width == 1500
    assert config.collage.height == 500
    assert config.collage.columns == 3
    assert config.collage.margin == 40
    assert config.collage.background == "#EFEFEF"
    assert config.collage.divider_width == 6
    assert config.collage.divider_color == "#123456"
    assert config.collage.jpeg_quality == 82
