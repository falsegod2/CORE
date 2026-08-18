from pathlib import Path
import re

root = Path(__file__).resolve().parent
cfg = (root / "configs.yaml").read_text()
models = (root / "models.py").read_text()
mod = (root / "s_multistep_consistency.py").read_text()

checks = {
    "inverse disabled": bool(re.search(r"inverse_loss_scale:\s*0(?:\.0+)?\b", cfg)),
    "S-aff enabled": bool(re.search(r"affordance_s_scale:\s*1(?:\.0+)?\b", cfg)),
    "S5A scale remains 0.05": bool(re.search(r"loss_scale:\s*0\.05\b", cfg)),
    "RandomHorizon enabled": "random_horizon:" in cfg and "sample_count: 3" in cfg,
    "unbiased reweight enabled": "unbiased_reweight: true" in cfg,
    "balanced schedule is checkpointed": "random_horizon_schedule" in mod and "persistent=True" in mod,
    "selector counter is checkpointed": "random_horizon_step" in mod,
    "selector avoids torch randperm": "torch.randperm(" not in mod,
    "all legacy horizon metrics remain": all(x in mod for x in [
        "s_multistep_loss_h", "s_multistep_cosine_h", "s_multistep_valid_h", "s_multistep_target_raw_std_h"
    ]),
    "full legacy aggregate remains": 'metrics["s_multistep_loss"] = full_total.detach()' in mod,
    "actual random train loss is separately logged": "s_multistep_train_loss" in mod,
    "counterfactual metrics remain": "s_cf_margin_shuffle_h" in mod,
    "DVC metrics remain": "s_dvc_reanchored_consistency_cos_h" in mod,
    "diagnostics remain no-grad": "with torch.no_grad():" in mod,
    "models wires random config": "random_horizon_enabled" in models,
}
failed = False
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok
if failed:
    raise SystemExit(1)
print("S5A RandomHorizon static audit passed.")
