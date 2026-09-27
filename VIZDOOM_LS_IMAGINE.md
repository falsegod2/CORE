# LS-Imagine -> ViZDoom adaptation

## What is preserved

The core LS-Imagine algorithm is left intact: RGB+affordance encoder/decoder,
short-term imagination, learned jump head, special RSSM jump action, zoomed
pseudo-state branch, jumping-steps prediction, accumulated-reward prediction,
and mixed long/short imagination.

## What changes across environments

MineDojo's MineCLIP + U-Net concentration module is Minecraft-specific.  For
`vizdoom_my_way_home` and `vizdoom_deadly_corridor`, it is replaced by an
RGB-only green-goal teacher.  The teacher observes the same RGB frame as the
agent and produces a heatmap.  It never reads depth, labels, automap, game
variables, coordinates, or simulator state.

When a compact high-confidence green target is visible, the adapter creates a
*mild* crop-and-resize pseudo-frame (default zoom factor 1.35).  The original
LS-Imagine machinery then learns a special jump transition from the real state
to that pseudo-state.  `ScoreStorage` is kept unchanged and can pair the zoomed
semantic score with a later real observation to supervise jumping steps and
accumulated reward.

`basic`, `defend_center`, and `predict_position` are wired into the environment
adapter, but their default teacher is `none`.  They are useful for interface
smoke tests; they are **not** a full LS-Imagine semantic-transfer evaluation
until a task-equivalent teacher is added.

## Full and no-intrinsic variants

Full adapted LS-Imagine:

```bash
bash scripts/train_vizdoom.sh my_way_home 0 full
```

Clean no-intrinsic ablation:

```bash
bash scripts/train_vizdoom.sh my_way_home 0 nointrinsic
```

The no-intrinsic profile sets both the model loss scale and actor imagination
reward scale of the intrinsic head to zero.  It does **not** disable heatmaps,
zoomed pseudo-states, jump labels, special jump dynamics, jumping-steps,
accumulated-reward prediction, or the long-term branch.

## First run

```bash
pip install "vizdoom>=1.3.0,<2.0"
python scripts/smoke_test_vizdoom_ls.py --task my_way_home --steps 100
bash scripts/train_vizdoom.sh my_way_home 0 full
```

For direct comparison with the current Dreamer and Stable-NoIntrinsic
MyWayHome experiments, use:

```bash
bash scripts/train_vizdoom.sh my_way_home 0 nointrinsic
```

The ViZDoom profile defaults to 64x64 RGB, frame skip 4, native environment
reward, 300k environment decisions, evaluation every 20k, and 10 evaluation
episodes.
