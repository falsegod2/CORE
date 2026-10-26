#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ENV_PYTHON:-$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -n "${ENV_PYTHON:-}" ]]; then
    echo "ENV_PYTHON is not executable: $PYTHON_BIN" >&2
    exit 1
  fi
  PYTHON_BIN="$(command -v python)"
fi
exec "$PYTHON_BIN" -u "$ROOT/run_ls_imagine.py" "$@"
