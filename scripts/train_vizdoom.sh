#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

SCENARIO="${1:-my_way_home}"
shift || true

python expr.py \
  --configs vizdoom \
  --task "vizdoom_${SCENARIO}" \
  --logdir ./logdir \
  "$@"
