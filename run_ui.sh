#!/usr/bin/env bash
# Launch the stochastic-medium lighting experiment window.
set -euo pipefail
cd "$(dirname "$0")"

if ! python3 -c "import numpy, scipy, PIL, matplotlib" >/dev/null 2>&1; then
  echo "[run_ui] Installing Python dependencies into user site-packages…"
  python3 -m pip install --user -r requirements.txt
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
echo "[run_ui] Open http://${HOST}:${PORT}/ in your browser."
exec python3 -m stochastic_medium.cli serve --host "$HOST" --port "$PORT"
