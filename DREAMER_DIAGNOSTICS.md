# Detached DreamerV3 Internal Diagnostics

This build adds observation-only diagnostics for fair mechanism comparison with Stable V1.2.
They do **not** enter any loss, do **not** update Dreamer parameters, and deterministic
RSSM `sample=False` rollouts avoid consuming the global PyTorch RNG.

## Logged metrics

### Controllability / counterfactual dynamics
- `dreamer_multistep_cosine_h{1,2,4,8,15}`
- `dreamer_cf_margin_shuffle_h*`
- `dreamer_cf_margin_zero_h*`
- `dreamer_cf_response_gap_shuffle_h*`
- `dreamer_cf_response_gap_zero_h*`
- `dreamer_cf_valid_h*`
- `dreamer_reanchor_gain_h15`

These use the same replay convention as Stable: `action[:, t]` leads into state `t`, so
rollout from state `t` starts with `action[:, t+1]`.

### Standard Dreamer reward-model return quality
- `dreamer_return_corr_h*`
- `dreamer_return_mae_h*`
- `dreamer_return_pred_std_h*`
- `dreamer_return_target_std_h*`
- `dreamer_return_valid_h*`

This is **not** Stable's Outcome head. It evaluates the k-step discounted return implied by
Dreamer's ordinary reward head under a deterministic open-loop latent rollout.

### Detached task-relevance probe
- `dreamer_task_probe_acc`
- `dreamer_task_probe_margin`
- `dreamer_task_probe_active_classes`
- `dreamer_task_evidence_span`
- `dreamer_task_success_fraction`

ViZDoom stores `heatmap` and `task_score` from the same RGB-only teacher family used by the
Stable adapter. Dreamer's encoder remains `cnn_keys: '^image$'`; these fields are never model
inputs or targets. The probe builds deterministic within-batch nearest-centroid classes and
has no trainable parameters.

Do not compare `dreamer_task_probe_acc` numerically as if it were identical to Stable
`ProtoAcc`: it is a detached separability probe, not the trained Proto objective.

## Runtime

Diagnostics run every 16 model updates by default. Change:

```yaml
dreamer_diagnostics:
  every: 16
```

For wall-clock throughput benchmarks, disable them:

```yaml
dreamer_diagnostics:
  enabled: False
vizdoom_diagnostic_teacher_mode: 'none'
```

## Important

Start a **new logdir**. Old replay `.npz` files do not contain the new diagnostic-only
`heatmap` / `task_score` keys and should not be mixed with new replay files.
