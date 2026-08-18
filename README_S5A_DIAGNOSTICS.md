# ISO3 S-only 5A + Confidence/Action Diagnostics

Base: `ISO3-SAff-SOnly5A-NoInverse`.

Training objective is unchanged:

- inverse loss scale = 0
- S-affordance scale = 1
- Z adversary scale = 0
- S-only 5A scale = 0.05
- horizons = [1, 2, 4, 8, 15]

This variant only adds diagnostics. Diagnostic values are never added to
`total_loss`; all diagnostic rollouts are executed under `torch.no_grad()` and
with `sample=False`, so they do not consume the global torch RNG and do not
create a gradient path.

## 1. Counterfactual Action Diagnostics

From the same replay posterior S_t, the code deterministically rolls out:

1. the true replay action prefix;
2. a deterministic cyclic-shuffled prefix from another batch/start sample;
3. an all-zero action prefix (OOD reference only).

For h in {1,2,4,8,15}, important metrics are:

- `s_cf_true_target_cos_h*`
- `s_cf_shuffle_target_cos_h*`
- `s_cf_zero_target_cos_h*`
- `s_cf_margin_shuffle_h* = true_target_cos - shuffle_target_cos`
- `s_cf_margin_zero_h*`
- `s_cf_response_gap_shuffle_h* = 1 - cos(true_pred, shuffled_pred)`
- `s_cf_response_gap_zero_h*`

Interpretation:

- positive/increasing `margin_shuffle`: the correct action sequence predicts
  the actual future S better than a wrong but valid action sequence;
- increasing response gap: S dynamics reacts to action changes;
- response gap alone is insufficient: a model can be action-sensitive but
  wrong, therefore use it together with target margin.

The shuffled counterfactual is the primary metric. The zero-action vector is
not a valid one-hot MineDojo action and is retained only as an OOD sensitivity
reference.

## 2. Direct-vs-Composed Consistency Surrogate

Fast-LeWM can compare a direct action-prefix predictor against a decomposed
prefix prediction. ISO3 S5A does not have such a direct prefix predictor: it
uses the same one-step RSSM transition autoregressively. Splitting that exact
same deterministic rollout into two pieces would be algebraically identical
and give a meaningless perfect consistency score.

Therefore this implementation uses an architecture-appropriate surrogate:

- direct: fully open-loop deterministic rollout from posterior S_t to t+15;
- reanchored/composed: use the replay posterior S_{t+7} as an intermediate
  anchor, then roll the remaining actions to t+15.

Metrics:

- `s_dvc_reanchored_consistency_cos_h15`
- `s_dvc_reanchored_gap_h15`
- `s_dvc_reanchored_confidence_h15`
- `s_dvc_direct_target_cos_h15`
- `s_dvc_reanchored_target_cos_h15`
- `s_dvc_reanchor_gain_h15`

Interpretation:

- high consistency / low gap: terminal S is relatively insensitive to a real
  midpoint correction;
- positive large `reanchor_gain`: open-loop drift before the midpoint is
  materially hurting the terminal prediction;
- this metric is a rollout-confidence diagnostic, not a training loss and not
  a literal reproduction of Fast-LeWM self-consistency.
