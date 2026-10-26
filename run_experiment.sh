#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ENV_PYTHON:-$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  if [[ -n "${ENV_PYTHON:-}" ]]; then
    echo "ENV_PYTHON is not executable: $PYTHON" >&2
    exit 1
  fi
  PYTHON="$(command -v python)"
fi
exec "$PYTHON" -u "$ROOT/run_experiment.py" "$@"
