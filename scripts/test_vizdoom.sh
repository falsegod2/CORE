#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="${1:-my_way_home}"
SEED="${2:-0}"
VARIANT="${3:-full}"
CONFIGS=(vizdoom)
if [[ "$VARIANT" == "nointrinsic" || "$VARIANT" == "no_intrinsic" ]]; then
  CONFIGS+=(vizdoom_nointrinsic)
fi
python test.py --configs "${CONFIGS[@]}" --task "vizdoom_${TASK}" --seed "$SEED" --logdir ./logdir
