# Experiment 5D — Outcome-Aware Multi-Step RSSM Dynamics

This project preserves the original RGB-only DreamerV3/MineDojo training pipeline and adds one training-only objective to the existing unified RSSM.

It deliberately removes the two unsupported components exposed by 5B–5C:

- no independent action-prefix Transformer;
- no prefix-continuation head.

The auxiliary objective directly regularizes the same open-loop RSSM states used by Dreamer imagination with:

1. weak absolute posterior–prior consistency;
2. action-induced latent-delta consistency;
3. cumulative reward consistency through the original reward head;
4. pairwise return ranking;
5. an optional real-action versus shuffled-action margin.

Run the audit and tests first:

```bash
python scripts/audit_outcomeaware5d.py
pytest -q tests/test_multistep_consistency.py tests/test_outcome_aware_multistep.py
```

Main run:

```bash
bash ./scripts/train_5d.sh harvest_log_in_plains 0 0
```

Arguments are task suffix, seed, and physical GPU id. The script exposes the selected physical GPU and always uses `cuda:0` inside the process.
