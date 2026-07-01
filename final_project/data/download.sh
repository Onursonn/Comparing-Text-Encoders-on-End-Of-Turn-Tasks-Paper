#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"

if [[ ! -d "$ROOT/ftad" ]]; then
  git clone --depth 1 https://github.com/alimehotline/FTAD.git "$ROOT/ftad"
  echo "Cloned FTAD to $ROOT/ftad"
else
  echo "FTAD already present at $ROOT/ftad"
fi

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "Creating venv and installing deps..."
  python3 -m venv "$PROJECT_ROOT/.venv"
  "$PROJECT_ROOT/.venv/bin/pip" install -q -r "$PROJECT_ROOT/requirements.txt"
fi

"$PROJECT_ROOT/.venv/bin/python" "$ROOT/prefetch_hf.py"
