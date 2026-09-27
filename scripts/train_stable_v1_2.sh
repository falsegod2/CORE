#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./scripts/train_stable_v1_2.sh <task_name> [seed]"
  echo "Example: ./scripts/train_stable_v1_2.sh harvest_log_in_plains 0"
  exit 1
fi

TASK="$1"
SEED="${2:-0}"

export MINEDOJO_HEADLESS=1
python expr.py \
    --configs minedojo stable_v1 stable_v1_1 stable_v1_2 \
    --task minedojo_${TASK} \
    --seed ${SEED} \
    --logdir ./logdir/stable_v1_2
