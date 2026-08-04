# Experiment 5E: Stability-First Outcome-Aware Multi-Step RSSM

5E is the next controlled experiment after the 5D-full and 5D-no-margin runs.
Those runs showed useful pre-collapse signals (positive delta cosine, action gap,
and return correlation) but severe model-gradient explosion. 5E therefore tests
whether the core outcome-aware hypothesis survives after removing the unstable
training paths.

## Exact changes from 5D

1. The auxiliary RSSM rollout uses `sample=True`, restoring the standard
   categorical straight-through gradient through probabilities instead of the
   deterministic `mode()` identity-through-logits estimator.
2. The auxiliary rollout saves and restores CPU/CUDA RNG states, so it does not
   perturb the main Dreamer random stream.
3. Action margin is removed from training. Action gap remains a diagnostic.
4. Return ranking is removed from training. Ranking accuracy remains a
   diagnostic and no small-standard-deviation division enters the loss graph.
5. Horizon 15 is removed. The first stability run trains only 1/2/4/8 steps.
6. Auxiliary weights are reduced:
   - absolute: 0.0025
   - delta: 0.005
   - return: 0.005
7. Auxiliary gradients are measured on their exact parameter support and scaled
   so that their norm is at most 10% of the base Dreamer gradient norm and at
   most 25 in absolute norm.
8. Curriculum starts later and ramps more slowly.

## Training objective

The trained objective is

    L = L_Dreamer + alpha * (
          0.0025 L_abs + 0.005 L_delta + 0.005 L_return
        )

where `alpha` is the dynamic gradient-balancing multiplier. Action gap and
return-ranking accuracy are logged but do not contribute to the gradient.

## Audit and tests

```bash
python scripts/audit_outcomeaware5e.py
pytest -q \
  tests/test_multistep_consistency.py \
  tests/test_outcome_aware_multistep.py \
  tests/test_outcome_aware_world_model_integration.py \
  tests/test_outcome_aware_stable5e.py
```

## Run

```bash
bash ./scripts/train_5e.sh harvest_log_in_plains 0 0
```

Arguments are task name, seed, and physical GPU ID.

## Ablations

Disable dynamic gradient balancing:

```bash
bash ./scripts/train_5e_nobalance.sh harvest_log_in_plains 0 0
```

Remove return supervision:

```bash
bash ./scripts/train_5e_noreturn.sh harvest_log_in_plains 0 0
```

## Mandatory early-stop checks

Inspect at 100k, 200k, 300k, and 400k. Stop the run if any of these persists:

- `model_grad_norm` is non-finite;
- `oa_aux_grad_finite` becomes 0;
- KL or image loss rises sharply for two consecutive model logs;
- `oa_aux_grad_scale` remains near 0 for a long interval;
- Action gap and return Pearson collapse while base model loss rises.

Primary stability metrics:

- `oa_base_grad_norm_on_aux_support`
- `oa_aux_grad_norm`
- `oa_aux_to_base_grad_ratio`
- `oa_aux_grad_scale`
- `oa_aux_base_grad_cosine`
- `oa_aux_grad_finite`

Primary mechanism metrics:

- `oa_delta_cosine_h4`, `oa_delta_cosine_h8`
- `oa_action_gap_h4`, `oa_action_gap_h8`
- `oa_return_pearson_h4`, `oa_return_pearson_h8`
- `oa_rank_accuracy_h4`, `oa_rank_accuracy_h8`

5E is a stability experiment, not yet a claimed final method. Horizon 15 and
ranking/margin losses should not be reintroduced until the 1--8 step version is
stable through at least 500k environment steps.
