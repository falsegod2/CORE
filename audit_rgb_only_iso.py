#!/usr/bin/env python3
"""Static audit for ISO3-RGB-Aux-NoLong."""
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parent
config = (ROOT / "configs.yaml").read_text(encoding="utf-8")
models = (ROOT / "models.py").read_text(encoding="utf-8")
networks = (ROOT / "networks.py").read_text(encoding="utf-8")

checks = {
    "encoder is RGB-only": "cnn_keys: '^image$'" in config,
    "decoder still predicts image+heatmap": "cnn_keys: '^(image|heatmap)$'" in config,
    "inverse target uses action[:, 1:]": 'data["action"][:, 1:]' in models,
    "inverse uses cross entropy": "F.cross_entropy" in models,
    "reset transitions are masked": 'data["is_first"][:, 1:]' in models,
    "Z action adversary exists": "z_action_adversary" in models,
    "gradient reversal exists": "gradient_reverse" in models,
    "branch KL metrics exist": 'metrics["kl_s"]' in models and 'metrics["kl_z"]' in models,
    "per-branch free bits exist": "branch_free = free / 2.0" in networks,
    "10-episode evaluation": "eval_episode_num: 10" in config,
    "no long heads in config": all(
        token not in config
        for token in ("jump_head:", "jumping_steps_head:", "accumulated_reward_head:")
    ),
}

for path in (ROOT / "models.py", ROOT / "networks.py", ROOT / "expr.py"):
    py_compile.compile(str(path), doraise=True)

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

if failed:
    raise SystemExit("Audit failed: " + ", ".join(failed))
print("All static audits passed.")
