# Loop Telegram Bot

Производственный Telegram-бот (aiogram v3) для подбора очков с интеграцией каталога через Google Sheets CSV.

## Требования

- Python 3.10+
- SQLite

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Настройка окружения

Создайте `.env` на основе следующего шаблона и заполните значения:

```dotenv
BOT_TOKEN=123456:ABCDEF
SHEET_CSV_URL=https://docs.google.com/spreadsheets/d/e/2PACX-1vRT2CXRcmWxmWHKADYfHTadlxBUZ-R7nEX7HcAqrBo_PzSKYrCln4HFeCUJTB2q_C7asfwO7AOLNiwh/pub?output=csv
DAILY_TRY_LIMIT=7
REMINDER_HOURS=24
MOCK_TRYON=1
UPLOADS_ROOT=./uploads
RESULTS_ROOT=./results
CSV_FETCH_TTL_SEC=60
CSV_FETCH_RETRIES=3
# Будущие интеграции:
NANO_API_URL=
NANO_API_KEY=
DRIVE_PUBLIC_BASE_URL=
```

`SHEET_CSV_URL` можно не указывать, если используете значение по умолчанию. Настройки загружаются через Pydantic и `python-dotenv`.

## Как работает каталог

- `GoogleSheetCatalog` подтягивает CSV по ссылке публикации Google Таблицы.
- Ответ кэшируется на 60 секунд (можно настроить через `CSV_FETCH_TTL_SEC`), чтобы не ддосить источник.
- Для устойчивости реализован повтор запросов (до `CSV_FETCH_RETRIES` попыток) с экспоненциальной задержкой.
- Каждая карточка преобразует Google Drive `view`-ссылку к прямому URL (`/uc?export=view&id=...`) перед отправкой пользователю.
- Фильтрация идёт по колонке «Пол»; если моделей мало, добавляются «Унисекс».
- Кнопки «Подробнее о модели» открывают ссылку из таблицы.

## Запуск

```bash
python -m app.main
```

Бот использует long polling. При старте создаются директории `uploads/` и `results/`, а каталог очков подгружается из Google Sheets.

## Тесты

```bash
pytest
```

## Архитектура

- `app/services/catalog_google.py` — сервис каталога с кэшем и retry.
- `app/services/repository.py` — SQLite-хранилище пользователей и дневных лимитов.
- `app/services/tryon_mock.py` — mock-генерация результатов (белые изображения).
- `app/fsm.py` — конечный автомат aiogram, который обращается к каталогу и управляет UX.
- `app/utils/drive.py` — преобразование ссылок Google Drive.

## План развития

- Интеграция NanoBanana вместо mock-сервиса.
- Подключение реального стора для выдачи превью пользователю.
