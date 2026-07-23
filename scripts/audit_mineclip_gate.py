#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
checks = {
    "networks.py": [
        "class GatedMineCLIPEncoder",
        "semantic = obs[self._key].float().detach()",
        "rgb_embed + self._residual_scale * delta",
    ],
    "models.py": [
        "networks.GatedMineCLIPEncoder",
        "mineclip_fusion",
        "get_metrics",
    ],
    "envs/tasks/base/clip_reward.py": [
        "get_logits_and_embedding",
        "global_embedding = flat_video_feats[0].detach().float().cpu()",
        "self.model.requires_grad_(False)",
    ],
    "envs/tasks/base/clip_wrapper.py": [
        "mineclip_embedding",
        "get_logits_and_embedding",
    ],
    "envs/tasks/base/agent_wrapper.py": [
        '"mineclip_embedding"',
    ],
}
errors=[]
for rel, required in checks.items():
    text=(ROOT/rel).read_text()
    for token in required:
        if token not in text:
            errors.append(f"{rel}: missing {token}")

config=yaml.safe_load((ROOT/"configs.yaml").read_text())["defaults"]
fusion=config.get("mineclip_fusion", {})
if not fusion.get("enabled"):
    errors.append("mineclip_fusion.enabled is not true")
if fusion.get("input_dim") != 512:
    errors.append("mineclip_fusion.input_dim must be 512")
if config["encoder"]["cnn_keys"] != "^image$":
    errors.append("RGB encoder is no longer RGB-only")
for forbidden in ["heatmap", "zoomed_image", "jumping_steps", "accumulated_reward"]:
    for rel in ["models.py", "networks.py", "configs.yaml"]:
        if forbidden in (ROOT/rel).read_text().lower():
            errors.append(f"{rel}: forbidden mechanism remains: {forbidden}")

if errors:
    print("AUDIT FAILED")
    print("\n".join(f"- {e}" for e in errors))
    sys.exit(1)
print("AUDIT PASSED: RGB DreamerV3 + frozen global MineCLIP embedding + gated residual fusion.")
