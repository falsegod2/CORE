#!/usr/bin/env bash
set -euo pipefail
TASK_NAME="${1:?Usage: bash ./scripts/train_5e.sh <task_name> [seed] [gpu_id]}"
SEED="${2:-0}"
GPU_ID="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MINEDOJO_HEADLESS=1
python expr.py \
  --configs minedojo outcomeaware5e \
  --task "minedojo_${TASK_NAME}" --seed "$SEED" --device cuda:0 \
  --logdir "./logdir_outcomeaware5e/${TASK_NAME}/seed_${SEED}"
