#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.vizdoom import available_tasks, make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="my_way_home", choices=available_tasks())
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--no-intrinsic", action="store_true")
    args = p.parse_args()

    env = make_env(
        args.task,
        seed=args.seed,
        frame_skip=args.frame_skip,
        teacher_mode="auto",
        intrinsic_mode="none" if args.no_intrinsic else "score_improvement",
    )
    obs = env.reset()
    required = {
        "image", "heatmap", "jump", "is_zoomed", "is_calculated",
        "is_first", "is_last", "is_terminal", "reward_on_zoomed",
        "intrinsic", "intrinsic_on_zoomed", "score", "score_on_zoomed",
        "jumping_steps", "accumulated_reward", "zoomed_image",
        "heatmap_on_zoomed",
    }
    missing = required - set(obs)
    assert not missing, missing
    assert obs["image"].shape == (64, 64, 3), obs["image"].shape
    assert obs["heatmap"].shape == (64, 64), obs["heatmap"].shape

    print("env_id:", env.env_id)
    print("teacher_mode:", env.teacher_mode)
    print("intrinsic_mode:", env.intrinsic_mode)
    print("max_episode_steps:", env.max_episode_steps)
    print("action_n:", env.action_space.n)
    print("image:", obs["image"].shape, obs["image"].dtype)

    total_reward = 0.0
    visible = jumps = successes = 0
    for t in range(args.steps):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        total_reward += float(reward)
        visible += int(bool(obs["is_zoomed"]))
        jumps += int(bool(obs["jump"]))
        successes += int(bool(info.get("success", False)))
        if t < 8 or obs["is_zoomed"] or done:
            print(
                f"t={t+1:03d} r={float(reward): .5f} "
                f"score={float(obs['score']):.4f} "
                f"zoom={float(obs['score_on_zoomed']):.4f} "
                f"is_zoomed={int(obs['is_zoomed'])} jump={int(obs['jump'])} "
                f"intr={float(obs['intrinsic']):.6f}"
            )
        if done:
            obs = env.reset()
    env.close()
    print("total_reward:", total_reward)
    print("zoomed_frames:", visible, "jump_labels:", jumps, "success_events:", successes)
    print("LS-Imagine ViZDoom smoke test OK")


if __name__ == "__main__":
    main()
