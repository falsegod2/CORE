# DreamerV3 with Task-Relevant Object Tokens

This branch keeps the RGB-only, single-stream DreamerV3 world model and the
unchanged behavior reward:

```text
environment reward + MineCLIP reward
```

It changes only representation learning:

1. The trainable Dreamer CNN produces a 4x4 grid of RGB patch tokens.
2. Frozen MineCLIP provides the task-text embedding and the 512-D global video
   embedding already computed for reward scoring; no second visual forward pass
   is added.
3. Text and global video semantics rank the 16 RGB patches; the top 8 become the
   candidate set.
4. Four task- and state-conditioned object queries aggregate those candidates
   for two iterations.
5. A task-relevant scene token and the four object tokens are injected back into
   the RGB token grid through a small gated residual.
6. The object-token aggregate is weakly aligned with the detached MineCLIP video
   embedding, while coverage and diversity losses regularize binding.
7. The flattened encoder dimensionality passed to RSSM is unchanged.

This is not pure unsupervised Slot Attention. Candidate selection and object
initialization are conditioned by frozen Minecraft-domain semantics. The global
MineCLIP feature is not concatenated directly into RSSM; it serves as a detached
condition and teacher for the RGB-derived object tokens.

There is no heatmap, affordance reward, S/Z split, inverse dynamics, or
long-term jump.

## Train

```bash
bash ./scripts/train.sh harvest_log_in_plains
```

For a specified physical GPU and seed:

```bash
bash ./scripts/train_gpu.sh 2 harvest_log_in_plains 0
```

The default log root is `./logdir_task_object`, so this branch can run in
parallel with the global MineCLIP gate and task-patch branches.

## Audit and tests

```bash
python ./scripts/audit_task_object.py
python ./scripts/test_task_object_encoder.py
python ./scripts/test_world_model_task_object.py
```

Use a fresh log directory. Replay and checkpoints from the baseline, global
gate, and task-patch variants are not compatible because this branch stores both
`task_embedding` and `mineclip_embedding` and contains new object-token weights.
