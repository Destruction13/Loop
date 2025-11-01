from pathlib import Path

from app.config import SocialLink, load_config


def test_load_config_reads_allowed_env(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BOT_TOKEN=token123",
                "SHEET_CSV_URL=https://example.com/catalog.csv",
                "GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890/edit?gid=42",
                "LANDING_URL=https://landing.example.com",
                'PROMO_CODE="SPECIAL"',
                "DAILY_TRY_LIMIT=9",
                "CATALOG_ROW_LIMIT=15",
                "PICK_SCHEME=GENDER_AND_GENDER_ONLY",
                'SOCIAL_LINKS_JSON=[{"title":"One","url":"https://example.com/one"}]',
                "NANOBANANA_API_KEY=super-secret",
                "GOOGLE_SERVICE_ACCOUNT_JSON=custom_creds.json",
            ]
        ),
        encoding="utf-8",
    )

    for name in (
        "BOT_TOKEN",
        "SHEET_CSV_URL",
        "LANDING_URL",
        "SOCIAL_LINKS_JSON",
        "PROMO_CODE",
        "DAILY_TRY_LIMIT",
        "CATALOG_ROW_LIMIT",
        "PICK_SCHEME",
        "GOOGLE_SHEET_URL",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    config = load_config(str(env_file))

    assert config.bot_token == "token123"
    assert config.sheet_csv_url == "https://example.com/catalog.csv"
    assert config.catalog_sheet_id == "1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
    assert config.catalog_sheet_gid == "42"
    assert config.site_url == "https://landing.example.com"
    assert config.privacy_policy_url == "https://telegra.ph/Politika-konfidencialnosti-LOOV-10-29"
    assert config.promo_code == "SPECIAL"
    assert config.promo_contact_code == "SPECIAL"
    assert config.daily_try_limit == 9
    assert config.catalog_row_limit == 15
    assert config.pick_scheme == "GENDER_AND_GENDER_ONLY"
    assert config.social_links == (
        SocialLink(title="One", url="https://example.com/one"),
    )
    assert config.google_service_account_json == Path("custom_creds.json")
    assert (
        config.contacts_sheet_url
        == "https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890/edit?gid=42"
    )

    # Fixed defaults remain unchanged
    assert config.reminder_hours == 24
    assert config.csv_fetch_ttl_sec == 60
    assert config.csv_fetch_retries == 3
    assert config.batch_size == 2
    assert config.batch_layout_cols == 2
    assert config.collage.slot_width == 1080
    assert config.collage.output_format == "PNG"
    assert config.contact_reward_rub == 1000
    assert config.nanobanana_api_key == "super-secret"
    assert config.enable_leads_export is True
    assert config.enable_social_ad is True


def test_load_config_defaults_without_optional_env(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BOT_TOKEN=minimal",
                "NANOBANANA_API_KEY=hidden",
                "GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/EXAMPLE/edit?gid=0",
            ]
        ),
        encoding="utf-8",
    )

    for name in (
        "BOT_TOKEN",
        "SHEET_CSV_URL",
        "LANDING_URL",
        "SOCIAL_LINKS_JSON",
        "PROMO_CODE",
        "DAILY_TRY_LIMIT",
        "CATALOG_ROW_LIMIT",
        "PICK_SCHEME",
        "GOOGLE_SHEET_URL",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "NANOBANANA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    config = load_config(str(env_file))

    assert config.bot_token == "minimal"
    assert (
        config.sheet_csv_url
        == "https://docs.google.com/spreadsheets/d/EXAMPLE/edit?gid=0"
    )
    assert config.catalog_sheet_id == "EXAMPLE"
    assert config.catalog_row_limit is None
    assert config.site_url == "https://loov.ru/"
    assert config.promo_code == ""
    assert config.promo_contact_code == "CONTACT1000"
    assert config.daily_try_limit == 7
    assert config.pick_scheme == "UNIVERSAL"
    assert config.social_links == ()
    assert config.google_service_account_json is None
    assert (
        config.contacts_sheet_url
        == "https://docs.google.com/spreadsheets/d/EXAMPLE/edit?gid=0"
    )
    assert config.nanobanana_api_key == "hidden"
