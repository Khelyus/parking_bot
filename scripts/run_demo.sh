#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if grep -q 'TELEGRAM_BOT_TOKEN=replace-with-your-token' .env; then
  echo "Set TELEGRAM_BOT_TOKEN in .env before starting the bot."
  exit 1
fi

python -m parking_bot --demo
