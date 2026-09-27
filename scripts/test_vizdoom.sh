#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: bash scripts/test_vizdoom.sh <scenario> <checkpoint_dir> [extra args]"
  echo "Example: bash scripts/test_vizdoom.sh my_way_home ./logdir/vizdoom_my_way_home/seed_0/20260923T120000"
  exit 1
fi
SCENARIO="$1"
CHECKPOINT="$2"
shift 2

python test.py \
  --configs vizdoom \
  --task "vizdoom_${SCENARIO}" \
  --agent_checkpoint_dir "$CHECKPOINT" \
  --logdir ./eval_logdir \
  "$@"
