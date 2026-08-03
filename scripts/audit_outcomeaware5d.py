#!/usr/bin/env python3
from pathlib import Path
import sys

import ruamel.yaml as yaml

root = Path(__file__).resolve().parents[1]
required = [
    root / "outcome_aware_multistep.py",
    root / "models.py",
    root / "expr.py",
    root / "configs.yaml",
    root / "scripts" / "train_5d.sh",
    root / "tests" / "test_outcome_aware_multistep.py",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing files:\n" + "\n".join(missing))

models = (root / "models.py").read_text()
expr = (root / "expr.py").read_text()
module = (root / "outcome_aware_multistep.py").read_text()
loader = yaml.YAML(typ="safe")
with (root / "configs.yaml").open() as handle:
    config = loader.load(handle)
profile = config.get("outcomeaware5d", {})
oa = profile.get("outcome_aware_multistep", {})

checks = {
    "module import": "import outcome_aware_multistep" in models,
    "world-model call": "self._outcome_aware(" in models,
    "actual env-step tensor": "self._wm._train(data, env_step)" in expr,
    "profile exists": bool(profile),
    "original 5A disabled in 5D": profile.get("multi_step_consistency", {}).get("enabled") is False,
    "5D enabled": oa.get("enabled") is True,
    "late-weighted horizons": oa.get("horizon_weights") == [0.25, 0.5, 0.75, 1.0, 1.0],
    "continuation removed": "continuation_head" not in module,
    "no prefix transformer": "TransformerEncoder" not in module and "nn.Transformer" not in module,
    "existing reward head used": "reward_head(dynamics.get_feat(current_real))" in module,
    "episode masking": "valid_real = valid_real & (~first_prefix_flat" in module,
    "deterministic action shuffle": "torch.roll" in module,
    "curriculum": "horizon_start_steps" in module and "global_start_step" in module,
}
for name, ok in checks.items():
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
if not all(checks.values()):
    sys.exit(1)

# The objective must not own a trainable alternative latent model.
sys.path.insert(0, str(root))
from outcome_aware_multistep import OutcomeAwareMultiStepRSSM
objective = OutcomeAwareMultiStepRSSM(
    feat_dim=24,
    horizons=(1, 2),
    horizon_weights=(1.0, 1.0),
    horizon_start_steps=(0, 0),
    horizon_ramp_steps=(0, 0),
)
trainable = sum(parameter.numel() for parameter in objective.parameters())
print(f"[{'OK' if trainable == 0 else 'FAIL'}] auxiliary trainable parameters: {trainable}")
if trainable != 0:
    sys.exit(1)
print("Experiment 5D static audit passed.")
