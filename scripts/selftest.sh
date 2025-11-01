#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "[ERROR] Виртуальное окружение .venv не найдено. Запустите scripts/install.sh" >&2
  exit 1
fi

source "${PROJECT_ROOT}/.venv/bin/activate"
python "${PROJECT_ROOT}/manage.py" check
STATUS=$?
if [[ $STATUS -eq 0 ]]; then
  echo -e "\033[32mSelf-check завершён без критических ошибок.\033[0m"
else
  echo -e "\033[31mSelf-check завершён с ошибками.\033[0m"
fi
exit $STATUS
