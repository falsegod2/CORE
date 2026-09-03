#!/bin/bash
# Sequential template. Delete profiles you do not need before running.
set -e
TASK="${1:-harvest_log_in_plains}"
SEED="${2:-0}"
PROFILES=(
  ab_dreamer
  ab_dual
  ab_saff
  ab_s5a
  ab_outcome
  ab_proto
  ab_saff_s5a
  ab_saff_s5a_outcome
  ab_saff_s5a_proto
  ab_full
)
for PROFILE in "${PROFILES[@]}"; do
  echo "===== $TASK | $PROFILE | seed=$SEED ====="
  bash ./scripts/train_ablation.sh "$TASK" "$PROFILE" "$SEED"
done
