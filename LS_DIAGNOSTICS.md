# Detached LS-Imagine diagnostics

This build adds observation-only diagnostics for mechanism comparison against Dreamer and Stable V1.2. None of the new metrics enters the world-model, actor, or critic objective. The counterfactual rollouts use `sample=False`, so diagnostic logging does not consume the global PyTorch RNG stream.

## Shared metrics for fair comparison

The ViZDoom profile enables diagnostics every 16 model updates.

- `ls_multistep_cosine_h{1,2,4,8,15}`: deterministic open-loop RSSM future-state similarity to the replay posterior target.
- `ls_cf_margin_shuffle_h*`: true-action target cosine minus shuffled-action target cosine.
- `ls_cf_response_gap_shuffle_h*`: latent response difference between true and shuffled action prefixes.
- `ls_cf_valid_h*`: fraction of valid windows that do not cross episode resets.
- `ls_reanchor_gain_h15`: improvement after re-anchoring the rollout at the midpoint posterior.
- `ls_return_corr_h*`: correlation between the standard LS reward-head multi-step return prediction and replay return.
- `ls_return_mae_h*`: corresponding multi-step return MAE.
- `ls_return_pred_std_h*`, `ls_return_target_std_h*`: predicted/target return dispersion.
- `ls_task_probe_acc`: detached nearest-centroid task-stage separability of the LS latent.
- `ls_task_probe_margin`: correct-class cosine margin over the strongest wrong class.
- `ls_task_evidence_span`: dynamic range of RGB-teacher evidence used only for the probe.

The task probe uses LS's existing ViZDoom `score` and `heatmap`. Since LS-Imagine intentionally feeds `image|heatmap` to its encoder, this probe measures the task-relevance structure of the actual LS latent; it is not an RGB-only probe.

## LS-specific long/short imagination metrics

- `ls_jump_positive_fraction`: replay fraction with positive jump labels.
- `ls_is_zoomed_fraction`: replay fraction with zoomed pseudo-observations.
- `ls_calculated_fraction`: replay fraction whose jump duration / accumulated reward target is valid.
- `ls_jump_pred_mean`, `ls_jump_pred_acc`: jump-head statistics.
- `ls_jumping_steps_mae`, `ls_jumping_steps_corr`: quality of the jump-duration head on valid zoom pairs.
- `ls_accumulated_reward_mae`, `ls_accumulated_reward_corr`: quality of the accumulated-reward head.
- `ls_imag_jump_fraction`: fraction of base imagined time-state pairs using long-term jump transitions.
- `ls_long_imag_sequence_fraction`: fraction of base imagined sequences that contain at least one long-term jump.
- `ls_imag_augmented_sequence_fraction`: fraction of final imagination sequences created by LS's post-jump augmentation.
- `ls_predicted_jumping_steps_mean_on_jump`: mean learned skip length on imagined transitions that actually jump.

## Invariance check

Run:

```bash
python scripts/test_ls_diagnostics.py
```

Expected:

```text
PASS: detached LS-Imagine diagnostics
PASS: global torch RNG unchanged
```

## Training

Use the same command as before, for example:

```bash
bash scripts/train_vizdoom.sh predict_position
```

or:

```bash
bash scripts/train_vizdoom.sh my_way_home
```

Use a fresh log directory for a clean comparison. No new Python dependency is introduced by the diagnostics themselves.
