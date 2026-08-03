# Experiment 5D: Outcome-Aware Multi-Step Latent Dynamics

## 1. Evidence-based motivation

5A showed that open-loop RSSM consistency can be optimized, but two seeds reached nearly identical multi-step cosine values while their control success differed substantially. Generic latent matching is therefore not a sufficient control objective.

5B showed a positive action gap, proving that a separate causal prefix predictor used the action sequence, but that branch did not improve the RSSM trajectory used by the actor and reduced control performance.

5C recovered control performance after adding prefix return supervision. Its continuation targets were approximately 99.99% positive and therefore provided almost no information. The useful candidate signal was return, not continuation.

5D consequently removes the separate prefix predictor and continuation head. All supervision is applied directly to the standard RSSM open-loop trajectory.

## 2. Objective

For posterior start state `s_t^q`, replay actions generate the deterministic auxiliary rollout

`ŝ_{t+k}^p = F_phi^(k)(stopgrad(s_t^q), a_{t+1:t+k})`.

`rollout_sample=false` uses the straight-through distribution mode. This avoids consuming the global random-number stream and makes the real-versus-shuffled comparison reproducible.

The objective is

`L_5D = rho(t) [0.005 L_abs + 0.020 L_delta + 0.010 L_return + 0.005 L_rank + 0.005 L_action]`.

### 2.1 Weak absolute consistency

`L_abs = 1 - cos(R f(ŝ_{t+k}^p), stopgrad(R f(s_{t+k}^q)))`.

This is retained only as a weak trajectory stabilizer.

### 2.2 Action-induced delta consistency

`Delta_pred = R f(ŝ_{t+k}^p) - R f(s_t^q)`

`Delta_target = R f(s_{t+k}^q) - R f(s_t^q)`

`L_delta = 1 - cos(Delta_pred, stopgrad(Delta_target))`.

Subtracting the start representation suppresses static background agreement and focuses supervision on action-induced change.

### 2.3 Open-loop cumulative reward consistency

At every imagined step, the original Dreamer reward head predicts `r_hat_{t+j}` from the imagined RSSM feature. The predicted prefix return is

`G_hat_{t,k} = sum_j gamma^(j-1) alive_{j-1} r_hat_{t+j}`.

It is matched to the replay prefix return in symlog space with Smooth-L1 loss. No independent return encoder or latent predictor is introduced.

Above-average replay returns receive bounded scale-free weights. This avoids a fixed threshold tied to one reward scale.

### 2.4 Pairwise return ranking

Cyclic deterministic pairs are formed inside the batch. When two replay prefixes have distinguishable returns, the predicted returns must have the same order. This tests decision relevance instead of absolute-value fitting alone.

### 2.5 Action margin

The same start state is rolled out with a deterministically shifted action prefix. The real action-induced delta should match the observed future delta by at least margin 0.1 more than the shuffled action delta. By default the shuffled branch is stop-gradient, preventing the auxiliary loss from deliberately corrupting counterfactual dynamics.

## 3. Curriculum

The actual environment step is passed from `expr.py` as a tensor.

- global auxiliary loss: starts at 50k and ramps to full strength by 300k;
- horizons 1/2/4: start at 50k;
- horizon 8: starts at 150k;
- horizon 15: starts at 300k;
- every horizon ramps for 100k steps.

This addresses the delayed learning observed in 5B and 5C without tying the schedule to optimizer update count.

## 4. Preserved experimental settings

The project retains the original MineDojo profile, RGB encoder, unified RSSM, reward/end heads, actor–critic, replay settings, evaluation interval, episode limit, task wrapper, and MineCLIP environment reward. No incompatible checkpoint should be loaded; use the new `logdir_outcomeaware5d` directory.

## 5. Principal diagnostics

- `oa_abs_cosine_h*`: absolute future-state agreement;
- `oa_delta_cosine_h*`: action-induced change agreement;
- `oa_action_gap_h*`: real versus shuffled action delta gap;
- `oa_return_mae_h*`, `oa_return_pearson_h*`: prefix-return quality;
- `oa_rank_accuracy_h*`: pairwise value ordering;
- `oa_horizon_gate_h*`, `outcome_aware_curriculum_scale`: schedule state;
- component contributions and `outcome_aware_total_loss`;
- standard evaluation success rate and successful episode length.

The primary claim requires multiple seeds. A single high-performing seed is not sufficient.
