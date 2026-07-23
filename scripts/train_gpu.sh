#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: ./scripts/train_gpu.sh <physical_gpu_id> <task_name> [seed]"
  exit 1
fi

GPU_ID="$1"
TASK_NAME="$2"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MINEDOJO_HEADLESS=1
python expr.py \
  --configs minedojo \
  --task "minedojo_${TASK_NAME}" \
  --seed "$SEED" \
  --device cuda:0 \
  --logdir ./logdir_task_patch
