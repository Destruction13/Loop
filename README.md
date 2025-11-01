<div align="center">
  <h1>LOOV Telegram Bot</h1>
  <h3>Telegram-бот, который подбирает оправы по фото и ведёт лиды в Google Sheets</h3>
</div>

<p align="center">
  <a href="https://t.me/LooV_TheGlassesBot"><img alt="Запустить бота" src="https://img.shields.io/badge/Запустить%20бота-Open%20in%20Telegram-1DA1F2?logo=telegram" /></a>
  <a href="#-быстрый-старт-локально"><img alt="Быстрый старт" src="https://img.shields.io/badge/Быстрый%20старт-CLI%20setup-43C59E" /></a>
  <a href="#-установка-на-vps-ubuntu-22042404"><img alt="Установка на VPS" src="https://img.shields.io/badge/Установка%20на%20VPS-Deploy-4C51BF" /></a>
  <a href="#-логи-и-мониторинг-ошибок"><img alt="Логи и диагностика" src="https://img.shields.io/badge/Логи%2FДиагностика-Inspect-805AD5" /></a>
  <a href="https://github.com/Destruction13/Loop/issues"><img alt="Issues" src="https://img.shields.io/badge/Issues-Обсудить-8A8D91?logo=github" /></a>
</p>

<video src="video/promo_start.MP4" controls playsinline style="max-width:100%; border-radius:12px;"></video>

<sub>⚠️ Если плеер не отобразился, <a href="video/promo_start.MP4">скачайте видео по прямой ссылке</a>.</sub>

[🚀 Быстрый старт](#-быстрый-старт-локально) · [🖥️ Установка](#-установка-на-vps-ubuntu-22042404) · [⚙️ Переменные](#-переменные-окружения) · [🧾 Логи](#-логи-и-мониторинг-ошибок) · [❓ FAQ](#-faq-и-типичные-ошибки)

## Оглавление
- [✨ Что это](#-что-это)
- [✅ Возможности](#-возможности)
- [🎬 Демонстрация](#-демонстрация)
- [🚀 Быстрый старт (локально)](#-быстрый-старт-локально)
- [🖥️ Установка на VPS (Ubuntu 22.04/24.04)](#-установка-на-vps-ubuntu-22042404)
- [⚙️ Переменные окружения](#-переменные-окружения)
- [🧾 Логи и мониторинг ошибок](#-логи-и-мониторинг-ошибок)
- [🩺 Диагностика (Self-check)](#-диагностика-self-check)
- [❓ FAQ и типичные ошибки](#-faq-и-типичные-ошибки)
- [📌 Roadmap](#-roadmap)
- [🔗 Полезные ссылки](#-полезные-ссылки)

## ✨ Что это
LOOV — это Telegram-бот на базе `aiogram`, `Pillow`, `SQLite` и Google Sheets API. Он анализирует фото пользователя, подбирает подходящие очки из каталога и отправляет варианты прямо в чат. Каталоги подгружаются из Google Sheets или CSV, генерация примерок выполняется через NanoBanana, а сам бот работает в режиме long polling — домен и вебхуки не нужны.

## ✅ Возможности
- Виртуальная примерка очков по пользовательскому фото.
- Одновременная выдача двух подходящих моделей для каждого запроса.
- Фильтрация каталога по полу и другим параметрам.
- Настраиваемые дневные лимиты попыток для каждого пользователя.
- Автоматический экспорт лидов и контактов в Google Sheets.
- Цветное логирование milestone-событий в консоли.
- Дублирование предупреждений и ошибок в отдельный лист Google Sheets.

## 🎬 Демонстрация
<video src="video/promo_start.mp4" controls playsinline style="max-width:100%; border-radius:12px;"></video>

Видео лежит в репозитории по пути [`video/promo_start.mp4`](video/promo_start.mp4).

## 🚀 Быстрый старт (локально)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# заполните значения переменных окружения

python -m app.main
```

## 🖥️ Установка на VPS (Ubuntu 22.04/24.04)
### 1. Системные пакеты
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-venv python3-pip ffmpeg chrony
```

### 2. Размещение проекта
```bash
sudo mkdir -p /opt/loov
sudo chown "$USER":"$USER" /opt/loov
cd /opt/loov
git clone https://github.com/Destruction13/Loop.git .
```

### 3. Виртуальное окружение и зависимости
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### 4. Конфигурация
```bash
cp .env.example .env
nano .env  # заполните токен, ссылки и ключи
```

### 5. Первый запуск и проверка
```bash
./scripts/selftest.sh
./scripts/run.sh  # убедитесь, что бот отвечает
```

### 6. Настройка systemd
```bash
sudo cp deploy/loov.service /etc/systemd/system/loov.service
sudo sed -i "s/<USERNAME>/$USER/g" /etc/systemd/system/loov.service
sudo systemctl daemon-reload
sudo systemctl enable loov
sudo systemctl start loov
```

### 7. Просмотр логов
```bash
journalctl -u loov -f -n 200
```

## ⚙️ Переменные окружения
| Переменная | Назначение | Пример |
| --- | --- | --- |
| `BOT_TOKEN` | Токен бота от @BotFather | `1234567890:ABCDEF...` |
| `SHEET_CSV_URL` или `GOOGLE_SHEET_URL` | Источник каталога: CSV-публикация или URL Google Sheets | `https://docs.google.com/.../pub?output=csv` |
| `SOCIAL_LINKS_JSON` | JSON с ссылками на соцсети (`[{"title":"VK","url":"https://vk.com/..."}]`) | `[{"title":"VK","url":"https://vk.com/loov"}]` |
| `NANOBANANA_API_KEY` | Ключ NanoBanana для генерации примерок | `nb_live_xxx` |
| `PROMO_CODE` | Промокод, который получит пользователь | `LOOV2024` |
| `DAILY_TRY_LIMIT` | Дневной лимит попыток для одного пользователя | `50` |
| `CATALOG_ROW_LIMIT` | Максимум строк каталога (0 = без лимита) | `0` |
| `PICK_SCHEME` | Схема подбора моделей (`UNIVERSAL`, `MALE`, `FEMALE` и т.д.) | `UNIVERSAL` |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Путь к JSON сервисного аккаунта | `service_account.json` |
| `LOG_SHEET_ID` | ID Google Sheets для логирования ошибок | `1AbCDeFgHiJkLmNoP` |
| `LOG_SHEET_NAME` | Название листа для логов | `Errors` |
| `ENABLE_SHEET_LOGGING` | Включить дублирование логов в таблицу (`0/1`) | `1` |

Пример `.env`:
```dotenv
BOT_TOKEN=""
SHEET_CSV_URL=""
SOCIAL_LINKS_JSON="[{\"title\":\"VK\",\"url\":\"https://vk.com/loov\"}]"
NANOBANANA_API_KEY=""
PROMO_CODE=""
DAILY_TRY_LIMIT=50
CATALOG_ROW_LIMIT=0
PICK_SCHEME="UNIVERSAL"
GOOGLE_SERVICE_ACCOUNT_JSON="service_account.json"
LOG_SHEET_ID=""
LOG_SHEET_NAME="Errors"
ENABLE_SHEET_LOGGING=1
```

## 🧾 Логи и мониторинг ошибок
- Основной поток — цветные milestone-события в stdout (`logger.py` использует Rich для подсветки).
- Записи уровней `WARNING`, `ERROR` и `CRITICAL` дублируются в Google Sheets (лист `LOG_SHEET_NAME`).
- Для корректной отправки логов настройте сервисный аккаунт Google:
  1. Создайте проект и сервисный аккаунт в Google Cloud Console, скачайте JSON-ключ.
  2. Поместите файл рядом с ботом и укажите путь в `GOOGLE_SERVICE_ACCOUNT_JSON`.
  3. Откройте таблицу Google Sheets с ID `LOG_SHEET_ID` и поделитесь ею с почтой сервисного аккаунта (доступ «Редактор»).
  4. Установите `ENABLE_SHEET_LOGGING=1`, перезапустите бота и убедитесь, что тестовые записи появляются в таблице.

## 🩺 Диагностика (Self-check)
- Скрипт `./scripts/selftest.sh` выполняет `python manage.py check` и проверяет ключевые зависимости.
- Валидирует `.env`, доступность CSV/Google Sheets, наличие `video/promo_start.mp4`, подключение к Telegram и права на файлы.
- Пример вывода:

```text
=== Self-check отчёт ===
✅ BOT_TOKEN: Токен Telegram-бота: OK
✅ DAILY_TRY_LIMIT: Лимит: 50
⚠️ Каталог CSV: Не удалось подключиться: HTTPSConnectionPool(...)
⚠️ Promo video: Файл отсутствует: /opt/loov/video/promo_start.mp4
❌ Google service account: Файл не найден по пути: /opt/loov/service_account.json
❌ Итог: обнаружены критические ошибки
```

Если отчёт завершился с ❌, исправьте конфигурацию и запустите проверку повторно.

## ❓ FAQ и типичные ошибки
### «Invalid JWT Signature»
- JSON-файл сервисного аккаунта повреждён или не полностью скопирован.
- В `.env` указан неверный путь к `GOOGLE_SERVICE_ACCOUNT_JSON`.
- Сервисному аккаунту не выдан доступ к таблице Google Sheets.
- На сервере некорректно настроено время: перезапустите `chrony` (`sudo systemctl restart chrony`).

### «promo_start.mp4 не найден»
- Убедитесь, что файл лежит по пути `video/promo_start.mp4`.
- Если видео не требуется, предупреждение можно игнорировать — бот продолжит работу в текстовом режиме.

### Проблемы с правами systemd
- Проверьте `User` и `Group` в `deploy/loov.service`.
- Передайте владение директорией `/opt/loov` и `service_account.json` нужному пользователю (`sudo chown -R`).
- Убедитесь, что виртуальное окружение и временные каталоги доступны сервису.

### Сбой синхронизации времени
- Проверьте статус `chrony`: `systemctl status chrony`.
- При необходимости выполните `sudo chronyc tracking` и дождитесь корректной синхронизации.

## 📌 Roadmap
- Подключить режим вебхуков с автоконфигурацией HTTPS.
- Добавить панель модерации для ручного выбора лучших кадров.
- Расширить отчётность по лидам в Google Data Studio.

## 🔗 Полезные ссылки
- 🤖 [Запустить бота в Telegram](https://t.me/LooV_TheGlassesBot)
- 🐙 [Issues проекта](https://github.com/Destruction13/Loop/issues)
- 📄 [Лицензия](LICENSE)
- ✉️ Контакты команды: `hello@loov.ru`
