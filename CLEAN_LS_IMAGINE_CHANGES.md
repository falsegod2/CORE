# Clean LS-Imagine baseline (Harvest Log in Plains)

This package is based on the supplied LS-Imagine source tree and applies the same clean evaluation/training protocol as the paired Clean DreamerV3 baseline.

## Changes

1. **Binary task reward** for `harvest_log_in_plains`: removed the additional custom `reward_specs: log: +1` from the train, rollout-collection, and test task definitions. The underlying MineDojo `harvest` task remains responsible for the task reward.
2. **No reward/state shortcut in the world-model encoder**: removed `RewardObs` from `expr.make_env` and disabled all MLP observation keys.
3. **Method-specific visual input is retained**: LS-Imagine still encodes and reconstructs `image` + `heatmap`, preserving the affordance mechanism.
4. **Training-time evaluation**: MineDojo override changed from 3 episodes every 10k steps to 10 episodes every 20k steps.
5. **Historical checkpoints**: added `checkpoint_every: 100000`; `checkpoint_100000.pt`, ..., are saved in addition to `latest.pt` when the step lands on the interval.
6. **Core LS-Imagine algorithm is unchanged**: long/short-term dynamics, jump prediction, affordance intrinsic reward, MineCLIP pathway, RSSM sizes, KL scales, optimizer settings, and imagination settings are preserved.

## First-run checks

For `harvest_log_in_plains`, startup should show no MLP encoder inputs, approximately:

```text
Encoder CNN shapes: {'image': (64, 64, 3), 'heatmap': (...)}
Encoder MLP shapes: {}
```

A successful episode should no longer receive the extra wrapper +1 task reward. For a 100-episode independent test, `eval_return` should be on the same order as `eval_success`, rather than approximately twice it, assuming the base MineDojo task emits the intended binary reward.

## Important

Do **not** resume an old checkpoint after these changes. The encoder input structure and task-reward protocol changed, so publication experiments should start from a fresh log directory.
