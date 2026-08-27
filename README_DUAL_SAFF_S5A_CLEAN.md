# Dual S/Z + S-Aff + S-only S5A — Clean ablation

This is the clean `Dual + S-Aff + S5A` ablation for comparison with the current
`BalancedOutcome-Clean` full model. It is built from the no-Outcome S5A codebase,
so Outcome code/head/loss/calibration are absent rather than merely assigned zero weight.

## Active
- Dual S/Z RSSM.
- S: action-conditioned controllable branch.
- Z: action-unconditioned contextual/residual branch.
- S-Aff: S-only affordance heatmap auxiliary supervision, scale 1.0.
- S-only multi-step action-conditioned latent prediction (S5A).
- Horizons: [1, 2, 4, 8, 15].
- S5A loss scale: 0.05.
- Counterfactual and re-anchoring S5A diagnostics.

## Disabled / absent
- Inverse loss = 0.0.
- Z-action adversarial loss = 0.0.
- Outcome / multi-step return prediction: absent.
- Long-term LS-Imagine branch: absent.

## Clean protocol inherited from the current full-model rerun
- Encoder: RGB only (`cnn_keys: ^image$`, no observation MLP branch).
- Heatmap is NOT an encoder input.
- Decoder reconstructs RGB + heatmap.
- S-Aff additionally reconstructs heatmap from S-only features.
- `RewardObs` is removed.
- Duplicate custom `reward_specs` for `harvest_log_in_plains` is removed.
- `eval_every = 20000`, `eval_episode_num = 10`.
- Historical checkpoints saved at each crossed ~100K environment-step boundary.

## Experimental purpose
Compare this package with `ISO3-SAff-SOnly5A-NoInverse-BalancedOutcome-Clean`
to isolate the incremental contribution of Balanced Outcome while keeping Dual,
S-Aff, S5A, observation protocol, task reward protocol, optimizer, and evaluation
settings otherwise aligned.

Do not resume a checkpoint from the Outcome-enabled model.
