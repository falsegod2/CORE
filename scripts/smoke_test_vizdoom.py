#!/usr/bin/env python3
"""Small environment-only smoke test; no GPU or Dreamer model required."""
import argparse
import sys
from pathlib import Path

import numpy as np

# When this file is executed as `python scripts/smoke_test_vizdoom.py`,
# Python puts `scripts/` rather than the repository root on sys.path.
# Add the root explicitly so sibling packages such as `envs/` are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.vizdoom import available_tasks, make_env


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="my_way_home", choices=available_tasks())
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--frame_skip", type=int, default=4)
args = parser.parse_args()

env = make_env(args.task, seed=0, frame_skip=args.frame_skip)
obs = env.reset()
print("env_id:", env.env_id)
print("image:", obs["image"].shape, obs["image"].dtype)
print("actions:", env.action_space.n)
ret = 0.0
for i in range(args.steps):
    action = np.random.randint(env.action_space.n)
    obs, reward, done, info = env.step(action)
    ret += float(reward)
    if done:
        print("episode done at", i + 1, "return", ret, "success", info["success"])
        obs = env.reset()
        ret = 0.0
print("smoke test OK")
env.close()
