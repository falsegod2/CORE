# ISO3 S-Aff + S-only 5A RandomHorizon + Diagnostics

Base: `ISO3-SAff-SOnly5A-NoInverse-Diagnostics`.

This version keeps the same model architecture and the same S5A scale (`0.05`),
keeps Inverse disabled and S-Aff enabled, and changes only the multi-horizon
training objective: each world-model update selects 3 of the 5 configured
horizons `{1,2,4,8,15}` to contribute gradient.

## 1. Balanced RandomHorizon schedule

Configuration:

```yaml
s_multi_step_consistency:
  horizons: [1, 2, 4, 8, 15]
  horizon_weights: [1.0, 1.0, 0.75, 0.5, 0.25]
  loss_scale: 0.05
  random_horizon:
    enabled: true
    sample_count: 3
    seed: 271828
    unbiased_reweight: true
```

For 5 horizons and sample_count=3 there are `C(5,3)=10` possible subsets.  A
local RNG object seeded with `271828` shuffles these 10 combinations once at
construction, and the resulting table is stored as a model buffer.  Training
cycles through that seed-defined pseudo-random order.

Therefore, during every complete 10-update cycle, every horizon is selected
exactly 6 times (inclusion probability `p=3/5=0.6`).  The selector counter and
schedule are checkpointed, so resume continues at the same position.

No PyTorch/NumPy/Python *global* RNG stream is consumed by horizon selection.
This prevents RandomHorizon itself from perturbing RSSM categorical sampling.

## 2. Expected S5A scale is preserved

The original fixed-horizon objective is

`L_full = sum_h w_h L_h / sum_h w_h`.

For a sampled subset, this version optimizes

`L_train = sum_h I_h * w_h * L_h / p / sum_h w_h`,

where `I_h` is 1 if horizon h is selected and `p=3/5`.

Thus `E[L_train] = L_full` under uniform subset coverage.  RandomHorizon changes
which temporal constraints act together on each update, rather than silently
reducing the average S5A coefficient.

## 3. All previous S5A metrics are preserved

The code still rolls the stochastic S prior to horizon 15 and computes every
legacy metric on every update, even if a horizon is not selected for gradient:

- `s_multistep_loss_h1/h2/h4/h8/h15`
- `s_multistep_cosine_h1/h2/h4/h8/h15`
- `s_multistep_valid_h1/h2/h4/h8/h15`
- `s_multistep_target_raw_std_h1/h2/h4/h8/h15`
- `s_multistep_loss`

For strict comparability, `s_multistep_loss` keeps its old meaning: the full
five-horizon weighted mean.  The loss actually used for backpropagation is
logged separately as:

- `s_multistep_train_loss`

RandomHorizon selection is logged as:

- `s_random_horizon_selected_h1/h2/h4/h8/h15` (0 or 1)
- `s_random_horizon_selected_count`
- `s_random_horizon_inclusion_prob`
- `s_random_horizon_selector_step`
- `s_random_horizon_enabled`
- `s_random_horizon_unbiased_reweight`

## 4. All previous diagnostics are retained

Counterfactual action diagnostics remain unchanged:

- `s_cf_true_target_cos_h*`
- `s_cf_shuffle_target_cos_h*`
- `s_cf_zero_target_cos_h*`
- `s_cf_margin_shuffle_h*`
- `s_cf_margin_zero_h*`
- `s_cf_response_gap_shuffle_h*`
- `s_cf_response_gap_zero_h*`
- `s_cf_valid_h*`

The posterior-reanchored Direct-vs-Composed surrogate also remains unchanged:

- `s_dvc_reanchored_consistency_cos_h15`
- `s_dvc_reanchored_gap_h15`
- `s_dvc_reanchored_confidence_h15`
- `s_dvc_direct_target_cos_h15`
- `s_dvc_reanchored_target_cos_h15`
- `s_dvc_reanchor_gain_h15`
- `s_dvc_valid_h15`

These diagnostics remain `torch.no_grad()` and are never added to `total_loss`.

## 5. What to compare against fixed S5A

The main scientific comparison is:

- fixed S5A: all `{1,2,4,8,15}` constrain S on every update;
- RandomHorizon: a balanced 3-of-5 subset constrains S on each update, with the
  same expected weighted S5A objective.

If RandomHorizon keeps the early rise in `imag_reward_std`, `value_std`, and
`actor_grad_norm` but avoids the late decline seen with fixed S5A, that supports
the hypothesis that simultaneous persistent multi-horizon regularization was
part of the late-stage over-constraint.

Useful curves to monitor together:

- `s_multistep_loss` vs `s_multistep_train_loss`
- `s_cf_margin_shuffle_h15`
- `s_cf_response_gap_shuffle_h15`
- `s_dvc_reanchored_gap_h15`
- `s_dvc_reanchor_gain_h15`
- `imag_reward_mean/std`
- `target_std`, `value_std`
- `actor_grad_norm`, `value_grad_norm`
- evaluation success/AUC

## 6. Tests

Run:

```bash
python test_s5a_random_horizon.py
python test_s5a_diagnostics_invariance.py
python audit_s5a_random_horizon.py
python audit_s5a_diagnostics.py
```
