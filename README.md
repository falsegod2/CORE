# DreamerV3 with Task-Conditioned Patch Tokens

This variant keeps the RGB-only, single-stream DreamerV3 world model and the
unchanged reward `environment + MineCLIP`. It adds one representation module:

- MineCLIP encodes the task text once and remains frozen.
- The trainable Dreamer CNN produces a 4x4 grid of RGB patch tokens.
- The task embedding scores the 16 patches.
- The top 4 task-relevant patches receive a conservative gated residual update.
- All 16 tokens are flattened into the same encoder dimensionality as baseline.

There is no global MineCLIP visual embedding, heatmap, affordance reward,
S/Z split, inverse dynamics, or long-term jump.

## Train

```bash
bash ./scripts/train.sh harvest_log_in_plains
```

The default log root is `./logdir_task_patch`, allowing this variant to run in
parallel with the global MineCLIP gate branch without sharing replay/checkpoints.

## Audit and tests

```bash
python ./scripts/audit_task_patch.py
python ./scripts/test_task_patch_encoder.py
python ./scripts/test_world_model_task_patch.py
```

Use a fresh log directory. Checkpoints and replay from the global-gate or plain
DreamerV3 variants are not compatible because `task_embedding` is a new field.
