#!/bin/bash
set -e

TASK="$1"
PROFILE="${2:-ab_dreamer}"
SEED="${3:-0}"

if [ -z "$TASK" ]; then
  echo "Usage: ./scripts/train_ablation.sh <task_name> [profile] [seed]"
  echo "Example: ./scripts/train_ablation.sh harvest_log_in_plains ab_saff_s5a_proto 0"
  echo "Profiles: ab_dreamer ab_dual ab_saff ab_s5a ab_outcome ab_proto ab_saff_s5a ab_saff_s5a_outcome ab_saff_s5a_proto ab_full"
  exit 1
fi

export MINEDOJO_HEADLESS=1
python expr.py \
  --configs minedojo "$PROFILE" \
  --task "minedojo_${TASK}" \
  --seed "$SEED" \
  --logdir "./logdir/${PROFILE}"
