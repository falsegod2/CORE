# Stable V1.2 -> ViZDoom

This port preserves the Stable V1.2 algorithm and adds ViZDoom as another
selectable environment suite.

## Main protocol

```bash
pip install "vizdoom>=1.3.0,<2.0"
python scripts/smoke_test_vizdoom_stable.py --task my_way_home --steps 100
bash scripts/train_vizdoom_stable_v1_2.sh my_way_home 0
```

Equivalent explicit command:

```bash
python expr.py \
  --configs vizdoom stable_v1 stable_v1_1 stable_v1_2 \
  --task vizdoom_my_way_home \
  --seed 0 \
  --logdir ./logdir/stable_v1_2_vizdoom
```

The default ViZDoom profile is **no intrinsic reward** and matches the Dreamer
proxy protocol: 64x64 RGB, frame_skip=4, native scenario reward, 10 evaluation
episodes, 20k evaluation interval, and 300k environment steps.

## What is preserved

- Dual S/Z RSSM
- RGB-only encoder
- S-only affordance supervision (S-Aff)
- S5A multi-step action-prefix consistency
- Balanced Outcome + calibration + evidence gate
- prototype utility + reliability gate + gradient guard
- success-aware replay reservoir
- actor/critic and imagination horizon

## Environment-specific teacher

Stable requires `heatmap` and `task_score` as training-only task-evidence
signals. MineDojo obtained these from its Minecraft-specific semantic/affordance
pipeline. For `vizdoom_my_way_home`, this port replaces that environment-specific
module with a deterministic RGB-only `green_goal` teacher:

- input: the same RGB frame observed by the agent;
- output: a soft spatial green-goal heatmap and a global visibility score;
- NOT fed to the RGB encoder;
- NOT added to actor reward;
- uses no depth, labels, automap, game variables, position or simulator state.

This keeps the algorithmic role of task-conditioned supervision while removing
MineCLIP/Minecraft dependencies. For fair LS-Imagine comparisons, reuse this
same teacher implementation.

`auto` selects `green_goal` for MyWayHome/DeadlyCorridor and `none` for tasks
where the target is not the green vest. Those latter tasks are runnable but do
not constitute a full task-evidence Stable comparison until a task-appropriate
teacher is defined.

## Switch tasks

```bash
bash scripts/train_vizdoom_stable_v1_2.sh my_way_home
bash scripts/train_vizdoom_stable_v1_2.sh deadly_corridor
bash scripts/train_vizdoom_stable_v1_2.sh defend_center
bash scripts/train_vizdoom_stable_v1_2.sh basic
bash scripts/train_vizdoom_stable_v1_2.sh predict_position
```

For the paper/proxy benchmark, start with `my_way_home`.
