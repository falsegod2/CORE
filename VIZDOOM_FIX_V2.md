# Stable V1.2 ViZDoom semantic fix V2

This patch fixes the two cross-environment semantic mismatches identified from
Crafter vs ViZDoom logs while preserving the original Stable V1.2 core.

## 1. Outcome positive evidence

Original Stable/Crafter behavior is retained by default:

- `positive_mode: absolute_nonzero`
- positive iff `abs(k-step return) > threshold`

ViZDoom overrides this with:

- `positive_mode: positive_return`
- positive iff `k-step return > threshold`

This prevents MyWayHome/PredictPosition living penalties from making nearly
every replay window a positive Outcome example. The Outcome head still predicts
the full native discounted return; only the balancing/evidence semantics change.

New diagnostics:

- `s_outcome_target_positive_frac_h*`
- `s_outcome_positive_mode_positive_return`
- `health/outcome_positive15`

## 2. Proto reliability is environment-scale aware

MineDojo/Crafter keep the original raw-span reliability:

- `reliability_mode: absolute_span`

ViZDoom uses:

- `reliability_mode: relative_contrast`
- `reliability_span_floor: 0.02`

The new term measures robust q05-q95 separation relative to the signal's own
magnitude and multiplies it by a small absolute-strength factor. This allows a
0.05-0.15-range but discriminative RGB teacher to open the gate without letting
1e-4 numerical noise count as reliable evidence.

## 3. PredictPosition moving-target teacher

`vizdoom_teacher_mode: auto` now maps:

- MyWayHome -> `green_goal`
- DeadlyCorridor -> `green_goal`
- PredictPosition -> `moving_target`

`moving_target` uses RGB frames only. It:

1. compares consecutive frames;
2. estimates dominant global translation by phase correlation;
3. aligns the previous frame to suppress camera-motion residual;
4. extracts compact residual motion;
5. softly weights saturated pixels;
6. emits `heatmap` + `task_score` for S-Aff/Proto.

It does NOT use ViZDoom labels, depth, automap, coordinates, or game variables.
The actor/world-model encoder remains RGB-only.

## 4. Unchanged components

- Dual S/Z RSSM
- S-Aff loss
- S5A loss and diagnostics
- Balanced Outcome architecture and scale
- Proto architecture, EMA bank, Gradient Guard
- Success Replay Reservoir
- Actor/Critic and imagination
- MineDojo/Crafter default semantics

## Validation performed

- Original S-Outcome tests: PASS
- Stable V1 tests: PASS
- Stable V1.1 reliability tests: PASS
- Stable V1.2 gradient/replay tests: PASS
- S5A invariance/RNG tests: PASS
- Python compileall: PASS
- shell syntax checks: PASS
- synthetic Outcome semantics: `[-.20, -.05, 0, +.8] -> 1 positive / 3 background`
- synthetic small-range teacher: relative reliability gate warms to 1
- synthetic moving-target RGB test: non-zero localized heatmap after camera shift

## Recommended reruns

First:

```bash
python scripts/smoke_test_vizdoom_stable.py --task my_way_home --steps 200
python scripts/smoke_test_vizdoom_stable.py --task predict_position --steps 200
```

Then rerun from scratch (do not resume old checkpoints):

```bash
bash scripts/train_vizdoom_stable_v1_2.sh my_way_home 0
bash scripts/train_vizdoom_stable_v1_2.sh predict_position 0
```

For the first comparison, 300K environment interactions are sufficient. Compare
against Dreamer/LS using environment data count (`dataset_size`) rather than
post-budget optimizer/update step counts.
