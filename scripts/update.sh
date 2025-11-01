#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="loov"

if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

SERVICE_PRESENT=0
if $SUDO systemctl list-unit-files --type=service | grep -q "${SERVICE_NAME}\.service"; then
  SERVICE_PRESENT=1
  $SUDO systemctl stop "${SERVICE_NAME}" || true
fi

git -C "${PROJECT_ROOT}" pull --ff-only

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  python3 -m venv "${PROJECT_ROOT}/.venv"
fi

"${PROJECT_ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${PROJECT_ROOT}/.venv/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ ${SERVICE_PRESENT} -eq 1 ]]; then
  $SUDO systemctl start "${SERVICE_NAME}"
else
  cat <<EOF
[INFO] systemd юнит ${SERVICE_NAME}.service не найден.
Чтобы установить сервис впервые:
  ${SUDO}cp deploy/loov.service /etc/systemd/system/loov.service
  ${SUDO}systemctl daemon-reload
  ${SUDO}systemctl enable --now loov
EOF
fi
