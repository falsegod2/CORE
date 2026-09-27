# ViZDoom changes

- Added `envs/vizdoom.py` with Gymnasium -> legacy Gym compatibility.
- Added `vizdoom` and `vizdoom_nointrinsic` config profiles.
- Added a configurable `intrinsic_reward_scale` in actor imagination; default
  is 1.0, so existing MineDojo behavior is unchanged.
- Added MyWayHome/DeadlyCorridor RGB green-goal affordance teacher and mild
  zoomed pseudo-state generation.
- Kept LS-Imagine jump/long-term machinery active in the no-intrinsic profile.
- Added smoke/train/test scripts and ViZDoom dependency entries.
