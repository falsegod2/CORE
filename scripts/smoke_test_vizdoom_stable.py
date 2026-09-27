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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="my_way_home", choices=available_tasks())
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()

    env = make_env(
        args.task,
        seed=args.seed,
        frame_skip=args.frame_skip,
        image_size=64,
        reward_mode="native",
        teacher_mode="auto",
    )
    obs = env.reset()
    print("env_id:", env.env_id)
    print("teacher_mode:", env.teacher_mode)
    print("max_episode_steps:", env.max_episode_steps)
    print("action_n:", env.action_space.n)
    print("image:", obs["image"].shape, obs["image"].dtype)
    print("heatmap:", obs["heatmap"].shape, obs["heatmap"].dtype,
          "min/max", int(obs["heatmap"].min()), int(obs["heatmap"].max()))
    print("task_score:", obs["task_score"])
    print("intrinsic:", obs["intrinsic"])

    total = 0.0
    success = False
    teacher_active = 0
    teacher_score_sum = 0.0
    teacher_score_max = 0.0
    heatmap_peak = 0
    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total += float(reward)
        success = success or bool(info.get("success", False))
        score = float(np.asarray(obs["task_score"]).reshape(-1)[0])
        peak = int(np.max(obs["heatmap"]))
        teacher_score_sum += score
        teacher_score_max = max(teacher_score_max, score)
        heatmap_peak = max(heatmap_peak, peak)
        teacher_active += int(peak > 0)
        if done:
            obs = env.reset()
    print("steps:", args.steps, "total_reward:", total, "success_seen:", success)
    print(
        "teacher_active_rate:", teacher_active / max(args.steps, 1),
        "task_score_mean/max:", teacher_score_sum / max(args.steps, 1), teacher_score_max,
        "heatmap_peak:", heatmap_peak,
    )
    env.close()
    print("Stable V1.2 ViZDoom smoke test OK")


if __name__ == "__main__":
    main()
