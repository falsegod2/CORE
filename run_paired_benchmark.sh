#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR_PYTHON="${ENV_PYTHON:-$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python}"
if [[ ! -x "$PAIR_PYTHON" ]]; then
  if [[ -n "${ENV_PYTHON:-}" ]]; then
    echo "ENV_PYTHON is not executable: $PAIR_PYTHON" >&2
    exit 1
  fi
  PAIR_PYTHON="$(command -v python)"
fi
exec "$PAIR_PYTHON" -u "$ROOT/run_paired_benchmark.py" "$@"
