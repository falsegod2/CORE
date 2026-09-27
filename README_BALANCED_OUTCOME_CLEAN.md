# Clean Balanced S-Outcome

This revision keeps the original Dual S/Z + S-Aff + S-only multi-step dynamics design, but fixes two experimental-protocol issues and one S-Outcome learning bottleneck.

## 1. Clean visual protocol
- Encoder: RGB only (`mlp_keys: '$^'`, `cnn_keys: '^image$'`).
- Decoder: RGB + affordance heatmap, no vector-state reconstruction.
- `RewardObs` is no longer attached to the environment.
- The affordance heatmap remains a **training target**, not an encoder input.

## 2. Binary harvest-log reward
For `harvest_log_in_plains`, the extra custom `log:+1` reward wrapper entry was removed so that the MineDojo harvest task reward is not counted twice. After this change, a successful one-log episode should have return around 1 rather than around 2. Verify this before long runs.

## 3. Balanced Multi-Step Return Prediction
The original Outcome regression was dominated by zero-return targets. The new default uses `balance_mode: balanced`:
- compute mean regression loss on positive-return targets;
- compute mean regression loss on zero-return targets;
- when both exist, mix them with `positive_mix=0.5`.

This prevents one rare positive target from being diluted by hundreds of zero targets. If a batch/horizon contains no positive target, the loss safely falls back to the negative-target mean.

## 4. Dense posterior head calibration (head only)
The same replay batch is additionally scanned over all legal start positions for each configured horizon. The Outcome head predicts the real discounted return from `[S_t, S_{t+k}]` posterior features. These posterior features are **detached**, so this calibration term trains only the Outcome head and cannot create a reward shortcut into the encoder/RSSM.

The original rollout Outcome path remains unchanged conceptually:
`[S_t, S_hat_{t+k}] -> predicted k-step return`.
This path still backpropagates through the predicted S future into the S dynamics.

Total Outcome auxiliary loss:
`L_out = L_rollout + calibration_scale * L_calibration`, with `calibration_scale=0.5`, then multiplied by the existing global `s_outcome_scale=0.01` in the world-model loss.

## 5. New diagnostics
Look for:
- `s_outcome_rollout_positive_count_h*`
- `s_outcome_rollout_negative_count_h*`
- `s_outcome_rollout_positive_loss_h*`
- `s_outcome_rollout_negative_loss_h*`
- `s_outcome_calibration_loss`
- `s_outcome_calibration_positive_count_h*`
- `s_outcome_calibration_negative_count_h*`
- existing `s_outcome_corr_h*`, `s_outcome_mae_h*`, `s_outcome_target_nonzero_frac_h*`

The key question is whether `s_outcome_corr_h15` rises meaningfully above the old ~0.03 level after reward-bearing replay becomes available.

## 6. Evaluation/checkpoints
- Training-time evaluation: 10 episodes every 20k steps.
- Historical checkpoint save: every 100k steps, in addition to `latest.pt`.

## First-run sanity checks
1. Startup should print `Encoder MLP shapes: {}`.
2. `harvest_log_in_plains` successful episode return should be about 1, not 2.
3. Run `python test_s_outcome.py` and `python audit_s_outcome.py` before training.
4. Start with one seed to 300-400k before launching all publication seeds.
