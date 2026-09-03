#!/bin/bash
TASK="$1"
PROFILE="${2:-ab_dreamer}"
SEED="${3:-0}"
if [ -z "$TASK" ]; then
  echo "Usage: ./scripts/train.sh <task_name> [ablation_profile] [seed]"
  exit 1
fi
exec bash ./scripts/train_ablation.sh "$TASK" "$PROFILE" "$SEED"
