#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "[ERROR] Виртуальное окружение .venv не найдено. Запустите scripts/install.sh" >&2
  exit 1
fi

source "${PROJECT_ROOT}/.venv/bin/activate"
python -m app.main
