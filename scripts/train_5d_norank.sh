#!/usr/bin/env bash
set -euo pipefail
TASK_NAME="${1:?Usage: bash ./scripts/train_5d_norank.sh <task_name> [seed] [gpu_id]}"
SEED="${2:-0}"
GPU_ID="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MINEDOJO_HEADLESS=1
python expr.py \
  --configs minedojo outcomeaware5d outcomeaware5d_norank \
  --task "minedojo_${TASK_NAME}" --seed "$SEED" --device cuda:0 \
  --logdir "./logdir_outcomeaware5d_norank/${TASK_NAME}/seed_${SEED}"
