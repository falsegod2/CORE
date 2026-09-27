from pathlib import Path
import re

root = Path(__file__).resolve().parent
cfg = (root / "configs.yaml").read_text()
models = (root / "models.py").read_text()
mod = (root / "s_multistep_consistency.py").read_text()

checks = {
    "inverse disabled": bool(re.search(r"inverse_loss_scale:\s*0(?:\.0+)?\b", cfg)),
    "S-aff enabled": bool(re.search(r"affordance_s_scale:\s*1(?:\.0+)?\b", cfg)),
    "S5A scale unchanged at 0.05": bool(re.search(r"loss_scale:\s*0\.05\b", cfg)),
    "diagnostics enabled": "diagnostics:" in cfg and "counterfactual_actions: true" in cfg,
    "counterfactual metrics exist": "s_cf_margin_shuffle_h" in mod,
    "reanchored consistency metrics exist": "s_dvc_reanchored_consistency_cos_h" in mod,
    "diagnostics use no_grad": "with torch.no_grad():" in mod,
    "diagnostics use deterministic mode": "sample=False" in mod,
    "shuffle does not use randperm": "torch.randperm(" not in mod,
    "diagnostics not added to total loss": "s_cf_" not in models and "s_dvc_" not in models,
}

failed = False
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok
if failed:
    raise SystemExit(1)
print("S5A diagnostics static audit passed.")
