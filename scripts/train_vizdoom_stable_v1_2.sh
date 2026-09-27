#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${1:-my_way_home}"
if [ "$#" -gt 0 ]; then shift; fi
SEED=0
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  SEED="$1"
  shift
fi

python expr.py \
  --configs vizdoom stable_v1 stable_v1_1 stable_v1_2 \
  --task "vizdoom_${TASK}" \
  --seed "$SEED" \
  --logdir ./logdir/stable_v1_2_vizdoom \
  "$@"
