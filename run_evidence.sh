#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evidence_python="${ENV_PYTHON:-$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python}"
if [[ ! -x "$evidence_python" ]]; then
  if [[ -n "${ENV_PYTHON:-}" ]]; then
    echo "ENV_PYTHON is not executable: $evidence_python" >&2
    exit 2
  fi
  evidence_python="$(command -v python)"
fi
exec "$evidence_python" -u "$project_root/run_evidence.py" "$@"
