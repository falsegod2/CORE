#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash ./scripts/train_repro.sh <task_name> [seed] [gpu_id]"
  exit 1
fi

TASK_NAME="$1"
SEED="${2:-0}"
GPU_ID="${3:-0}"

# These must be set before Python imports torch/CUDA.
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MINEDOJO_HEADLESS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python expr.py \
  --configs minedojo reproducible \
  --task "minedojo_${TASK_NAME}" \
  --seed "$SEED" \
  --device cuda:0 \
  --compile False \
  --deterministic_run True \
  --logdir "./logdir_ls_imagine_repro"
