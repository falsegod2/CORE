from pathlib import Path
import re

root = Path(__file__).resolve().parent
cfg = (root / "configs.yaml").read_text()
models = (root / "models.py").read_text()
mod = (root / "s_multistep_consistency.py").read_text()

checks = {
    "inverse disabled": bool(re.search(r"inverse_loss_scale:\s*0(?:\.0+)?\b", cfg)),
    "S-aff enabled": bool(re.search(r"affordance_s_scale:\s*1(?:\.0+)?\b", cfg)),
    "fixed S5A scale 0.05": "loss_scale: 0.05" in cfg,
    "S-Outcome enabled": "outcome:" in cfg and "enabled: true" in cfg,
    "S-Outcome scale 0.01": "loss_scale: 0.01" in cfg,
    "real replay reward passed": 'rewards=data["reward"]' in models,
    "S-Outcome enters model loss": "self._s_outcome_scale * s_outcome_loss" in models,
    "S-Outcome not added to actor reward": "s_outcome" not in models[models.find("class ImagBehavior"):],
    "uses predicted S future": "self.outcome_head(" in mod and "pred_feat" in mod,
    "uses discounted environment return": "discounted_return" in mod and "reward_step" in mod,
    "symlog target": "target_outcome_symlog = self._symlog(target_return)" in mod,
    "local RNG initialization": "torch.random.fork_rng(devices=[])" in mod,
    "counterfactual diagnostics retained": "s_cf_margin_shuffle_h" in mod,
    "reanchor diagnostics retained": "s_dvc_reanchor_gain_h" in mod,
}

failed = False
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok
if failed:
    raise SystemExit(1)
print("S-Outcome static audit passed.")
