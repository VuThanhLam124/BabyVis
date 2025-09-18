#!/usr/bin/env bash
set -euo pipefail

# BabyVis bootstrap helper. Installs dependencies (if requested) and launches the web UI.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${BABYVIS_VENV:-}" # Optional path to Python virtualenv
PYTHON_BIN="${BABYVIS_PYTHON:-python3}"
INSTALL=0
PROVIDER="${BABYVIS_MODEL_PROVIDER:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      INSTALL=1
      shift
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "❌ Python interpreter '$PYTHON_BIN' not found." >&2
  exit 1
fi

if [[ $INSTALL -eq 1 ]]; then
  echo "📦 Installing BabyVis dependencies..."
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo "🍼 Launching BabyVis"
[[ -n "$PROVIDER" ]] && export BABYVIS_MODEL_PROVIDER="$PROVIDER"

exec "$PYTHON_BIN" main.py --mode web "$@"
