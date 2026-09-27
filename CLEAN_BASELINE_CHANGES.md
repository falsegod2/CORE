# Clean DreamerV3 baseline changes

This package is derived from the provided DreamerV3 MineDojo code and applies only the protocol-cleaning changes discussed for the paper rerun.

## Changes

1. **Strict pixel-only world-model observation**
   - `encoder.mlp_keys: '$^'`
   - `decoder.mlp_keys: '$^'`
   - active `wrappers.RewardObs` call removed from `expr.py`
   - RGB remains the only Dreamer encoder/decoder observation target.

2. **Binary task reward for `harvest_log_in_plains`**
   - removed the extra custom `reward_specs.item_rewards.log.reward: 1` wrapper reward.
   - the underlying MineDojo `harvest` task remains responsible for the task reward.
   - after rerunning, verify that a successful episode has return approximately 1 rather than 2.

3. **More reliable online evaluation**
   - `eval_every: 20000`
   - `eval_episode_num: 10`

4. **Historical checkpoints**
   - keeps `latest.pt` as before
   - additionally saves `checkpoint_100000.pt`, `checkpoint_200000.pt`, ... when those steps are reached.

## Intentionally unchanged

- `steps: 1e6`
- `batch_size: 16`, `batch_length: 32`, `train_ratio: 16`
- RSSM size (`deter=4096`, `stoch=32`, `discrete=32`, hidden/units=1024)
- `dyn_scale: 0.5`, `rep_scale: 0.1`, `kl_free: 1.0`
- `imag_horizon: 15`, `discount: 0.997`, `discount_lambda: 0.95`
- `mineclip_reward_scale: 1.0`
- `break_speed_multiplier: 100`

## First-run checks

At startup, the model should print:

```text
Encoder CNN shapes: {'image': (64, 64, 3)}
Encoder MLP shapes: {}
```

For `harvest_log_in_plains`, successful episodes should satisfy approximately:

```text
return ~= success
```

rather than the previous `return ~= 2 * success`.

Do **not** resume an old checkpoint: the encoder architecture and reward protocol changed. Start a fresh log directory.
