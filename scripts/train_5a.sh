#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash ./scripts/train_5a.sh <task_name> [seed] [gpu_id]"
  echo "Example: bash ./scripts/train_5a.sh harvest_log_in_plains 0 0"
  exit 1
fi

TASK_NAME="$1"
SEED="${2:-0}"
GPU_ID="${3:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MINEDOJO_HEADLESS=1

python expr.py \
  --configs minedojo multistep5a \
  --task "minedojo_${TASK_NAME}" \
  --seed "$SEED" \
  --device cuda:0 \
  --logdir "./logdir_multistep5a/${TASK_NAME}/seed_${SEED}"
