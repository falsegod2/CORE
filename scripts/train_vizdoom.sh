#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${1:-my_way_home}"
SEED="${2:-0}"
VARIANT="${3:-full}"
COUNT=$#
if [[ $COUNT -ge 3 ]]; then
  shift 3
elif [[ $COUNT -eq 2 ]]; then
  shift 2
elif [[ $COUNT -eq 1 ]]; then
  shift 1
fi

CONFIGS=(vizdoom)
case "$VARIANT" in
  full|intrinsic)
    ;;
  nointrinsic|no_intrinsic|no-int)
    CONFIGS+=(vizdoom_nointrinsic)
    ;;
  *)
    echo "Unknown variant: $VARIANT (use full or nointrinsic)" >&2
    exit 2
    ;;
esac

python expr.py \
  --configs "${CONFIGS[@]}" \
  --task "vizdoom_${TASK}" \
  --seed "$SEED" \
  --logdir ./logdir \
  "$@"
