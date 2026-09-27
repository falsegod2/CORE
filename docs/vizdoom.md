# ViZDoom support

This repository keeps the original MineDojo path and adds ViZDoom as a second
environment suite. The DreamerV3 world model, actor-critic, latent imagination,
and optimization code are unchanged.

## Install

In the same Python environment used for Dreamer:

```bash
pip install "vizdoom>=1.3.0,<2.0"
```

The project already depends on Gymnasium. Current ViZDoom releases register
Gymnasium environments when `vizdoom.gymnasium_wrapper` is imported.

## Supported scenarios

| task suffix | ViZDoom environment | purpose | Doom timeout |
|---|---|---|---:|
| `basic` | `VizdoomBasic-v1` | minimal aim/shoot sanity check | 300 tics |
| `my_way_home` | `VizdoomMyWayHome-v1` | sparse first-person navigation | 2100 tics |
| `defend_center` | `VizdoomDefendCenter-v1` | dynamic target / aiming | 2100 tics |
| `deadly_corridor` | `VizdoomDeadlyCorridor-v1` | navigation + combat + survival | 2100 tics |
| `predict_position` | `VizdoomPredictPosition-v1` | delayed action consequence | 300 tics |

The adapter automatically falls back to older `-v0` IDs when possible.

## Pixel-only protocol

Dreamer receives only:

```text
image: uint8 [64, 64, 3]
is_first
is_last
is_terminal
```

Depth, labels, automap, and game variables are deliberately hidden. A zero
`mineclip_reward` field is carried only because this codebase's clean Dreamer
world-model class still contains that optional head. Under the `vizdoom`
configuration:

```yaml
mineclip_reward_scale: 0.0
mineclip_head.loss_scale: 0.0
grad_heads: [decoder, reward, end]
```

Thus MineCLIP contributes neither to world-model learning nor actor return.

## Train and switch environments

```bash
bash scripts/train_vizdoom.sh my_way_home
bash scripts/train_vizdoom.sh defend_center
bash scripts/train_vizdoom.sh deadly_corridor
bash scripts/train_vizdoom.sh basic
bash scripts/train_vizdoom.sh predict_position
```

Equivalent direct command:

```bash
python expr.py --configs vizdoom --task vizdoom_my_way_home --logdir ./logdir
```

Extra config values can still be overridden:

```bash
bash scripts/train_vizdoom.sh my_way_home --steps 100000 --seed 1
```

## Frame skip

Default:

```yaml
vizdoom_frame_skip: 4
```

One Dreamer action therefore advances four Doom tics. `episode_max_steps` is
computed in agent decisions, e.g. MyWayHome: ceil(2100 / 4) = 525.

For a strict one-action-per-tic experiment:

```bash
bash scripts/train_vizdoom.sh my_way_home --vizdoom_frame_skip 1
```

Keep this value identical across Dreamer, LS-Imagine, and Stable V1.2.

## Reward mode

Default `native` uses the official scenario reward. An optional sparse protocol
is provided for controlled ablations:

```bash
bash scripts/train_vizdoom.sh my_way_home --vizdoom_reward_mode sparse_success
```

`sparse_success` emits +1 only on the first task-specific success event and 0
otherwise. Do not mix reward modes when comparing algorithms.

## Smoke test

Before a long run:

```bash
python scripts/smoke_test_vizdoom.py --task my_way_home --steps 100
```

## Evaluate a checkpoint

```bash
bash scripts/test_vizdoom.sh my_way_home <checkpoint_directory>
```

The checkpoint directory is the timestamp directory that contains `latest.pt`.
