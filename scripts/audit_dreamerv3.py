#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SCAN = [
    ROOT / "configs.yaml",
    ROOT / "models.py",
    ROOT / "networks.py",
    ROOT / "expr.py",
    ROOT / "tools.py",
    ROOT / "envs/tasks/base/agent_wrapper.py",
    ROOT / "envs/tasks/minedojo/__init__.py",
]
FORBIDDEN = [
    "stoch_s", "stoch_z", "deter_s", "deter_z",
    "inverse_loss", "_inverse_dynamics", "loss_inverse",
    "heatmap", "affordance", "zoomed_image", "observe_zoomed",
    "jumping_steps", "accumulated_reward", "lambda_return_for_ls_imagine",
]
errors = []
for path in SCAN:
    text = path.read_text().lower()
    for token in FORBIDDEN:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token.lower())}(?![A-Za-z0-9_])", text):
            errors.append(f"{path.relative_to(ROOT)} contains forbidden token: {token}")

networks = (ROOT / "networks.py").read_text()
models = (ROOT / "models.py").read_text()
config = (ROOT / "configs.yaml").read_text()
for required in [
    '"stoch":', '"deter":', 'class RSSM',
    'dyn_scale * dyn_loss + rep_scale * rep_loss',
]:
    if required not in networks:
        errors.append(f"Missing standard RSSM marker: {required}")
if 'metrics["prior_ent"]' not in models or 'metrics["post_ent"]' not in models:
    errors.append("Single-stream RSSM entropy metrics are missing")
if "cnn_keys: '^image$'" not in config:
    errors.append("Encoder/decoder are not explicitly RGB-only")
if "inverse_loss_scale" in config:
    errors.append("inverse_loss_scale remains in config")

if errors:
    print("AUDIT FAILED")
    print("\n".join(f"- {item}" for item in errors))
    sys.exit(1)
print("AUDIT PASSED: single-stream DreamerV3 RSSM; RGB-only; no S/Z, inverse, heatmap, or long-term path.")
