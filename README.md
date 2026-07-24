# DreamerV3 with Competitive Task-Relevant Object Tokens V2

This branch keeps the RGB-only, single-stream DreamerV3 agent and the unchanged
behavior reward:

```text
environment reward + MineCLIP reward
```

It is a structural correction of the first task-object-token branch. The V1
training logs showed near-complete collapse: one RGB patch received roughly all
relevance mass and all four object queries attended to the same region. V2
changes the binding mechanism rather than merely increasing the diversity-loss
coefficient.

## V2 representation path

1. The trainable Dreamer CNN produces a 4x4 grid of 16 RGB patch tokens.
2. Frozen MineCLIP provides a task-text embedding and the 512-D video embedding
   already computed for reward scoring; no additional MineCLIP visual forward
   pass is introduced.
3. Text and video semantics rank the RGB patches. The top 6 form the candidate
   set.
4. The sharp relevance distribution is blended 50/50 with a uniform prior over
   the selected candidates.
5. Two task- and state-conditioned object queries aggregate candidates through
   a balanced log-space Sinkhorn assignment. Each patch competes across objects,
   and each object receives balanced assignment mass.
6. A low-entropy competition loss encourages candidate patches to choose a
   specific object; weak attention and feature-diversity terms discourage the
   two objects from becoming identical.
7. Only the task-relevant global scene token is aligned with the detached global
   MineCLIP video embedding. Individual object tokens are no longer averaged
   toward the same semantic target.
8. Global and object information is written back to the original RGB patch grid
   through a conservative residual gate. RSSM input dimensionality is unchanged.

Default corrective settings:

```yaml
num_objects: 2
candidate_topk: 6
relevance_temperature: 0.5
relevance_bias: 0.25
assignment_uniform_mix: 0.5
sinkhorn_iters: 4
residual_scale: 0.03
```

There is no heatmap, affordance intrinsic reward, S/Z split, inverse dynamics,
pure unsupervised Slot Attention, or long-term jump.

## Train

```bash
bash ./scripts/train.sh harvest_log_in_plains
```

For a specified physical GPU and seed:

```bash
bash ./scripts/train_gpu.sh 3 harvest_log_in_plains 0
```

The default log root is `./logdir_task_object_v2`. Use a fresh replay and
checkpoint; V1 checkpoints are architecturally incompatible because V2 changes
both the number of object queries and the assignment/semantic heads.

## Audit and tests

```bash
python ./scripts/audit_task_object.py
python ./scripts/test_task_object_encoder.py
python ./scripts/test_world_model_task_object.py
```

## Critical diagnostics

The main success criterion is not reward alone. V2 should also show:

- `task_object_attention_overlap` substantially below 1;
- `task_object_feature_overlap` below 1 and preferably decreasing;
- `task_object_sinkhorn_row_error` and `...col_error` near zero;
- `task_object_usage_entropy` near `log(2) = 0.693`, showing both objects are
  used;
- `task_object_global_semantic_cosine` increasing without forcing object-token
  equality;
- `task_object_delta_ratio` remaining conservative during early training.
