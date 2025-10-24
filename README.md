# Loop Telegram Bot Skeleton

Производственный skeleton Telegram-бота под try-on очков. Архитектура позволяет заменить mock-сервисы на реальные интеграции без переписывания бизнес-логики.

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

Скопируйте `.env.example` в `.env` и заполните значения:

```bash
cp .env.example .env
```

Минимально необходимо указать `BOT_TOKEN`.

## Запуск

```bash
python -m app.main
```

Бот использует long polling (aiogram v3). При первом запуске будут созданы локальные каталоги для загрузок и результатов.

## Структура каталогов

- `catalog/` — локальный каталог моделей (вместо Google Drive). Каждая модель хранится в подпапке с `meta.json`, `.keep` и опциональным `overlay.png`. При отсутствии превью сервис сам сгенерирует белый квадрат и сохранит рядом `thumb_auto.jpg`.
- `uploads/` — пользовательские загрузки.
- `results/` — сгенерированные изображения (mock try-on).

Пример каталога: `catalog/male/25-34/normal/model-001/meta.json`.

## Тесты

```bash
pytest
```

## Реальные интеграции (TODO)

- `app/services/drive_future.py` — заготовка под Google Drive API (публикация и выдача ссылок).
- `app/services/tryon_nanobanana.py` — заготовка под NanoBanana API.

Реализации должны следовать существующим интерфейсам в `catalog_base.py`, `tryon_base.py`, `storage_base.py`.

## Makefile

Удобные команды запуска, тестов и линтинга находятся в `Makefile`.
