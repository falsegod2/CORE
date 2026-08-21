# DreamerV3 MineDojo Baseline

This branch restores the previous experimental code to a standard single-stream DreamerV3 architecture.

## Active architecture

- RGB observation only.
- One recurrent deterministic state `deter` and one stochastic state `stoch`.
- Standard DreamerV3 posterior and prior distributions.
- Standard dynamics and representation KL losses with stop-gradient.
- Standard 15-step latent imagination and lambda return.
- No S/Z latent split and no inverse-dynamics auxiliary loss.
- No heatmap, affordance model, affordance reward, zoom observation, or long-term latent jump.

## Reward setting

The default `minedojo` configuration follows the LS-Imagine experimental reward setting rather than a sparse-only DreamerV3 setting:

```text
environment task reward + MineCLIP progress reward
```

Thus the precise experiment name is **DreamerV3 + MineCLIP**. The world-model and actor-critic architecture are DreamerV3; MineCLIP is retained only as task reward shaping for comparison with LS-Imagine.

## Training

```bash
conda activate ls_imagine
bash ./scripts/train.sh harvest_log_in_plains
```

Use a new log directory. Checkpoints from the S/Z, heatmap, or LS-Imagine branches are incompatible.

## Static audit

```bash
python ./scripts/audit_dreamerv3.py
```
