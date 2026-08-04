#!/usr/bin/env python3
import pathlib
import sys
import types

sys.modules.setdefault("wandb", types.SimpleNamespace())
_tb = types.ModuleType("torch.utils.tensorboard")
class _DummyWriter:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return lambda *args, **kwargs: None
_tb.SummaryWriter = _DummyWriter
sys.modules.setdefault("torch.utils.tensorboard", _tb)

import ruamel.yaml as yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradient_balance  # noqa: F401
import outcome_aware_multistep  # noqa: F401

loader = yaml.YAML(typ="safe")
configs = loader.load((ROOT / "configs.yaml").read_text())
cfg = configs["outcomeaware5e"]["outcome_aware_multistep"]

checks = {
    "profile exists": "outcomeaware5e" in configs,
    "5E enabled": cfg["enabled"] is True,
    "horizons stop at 8": cfg["horizons"] == [1, 2, 4, 8],
    "stochastic rollout": cfg["rollout_sample"] is True,
    "main RNG preserved": cfg["preserve_rng_state"] is True,
    "action margin removed": float(cfg["action_margin_loss_scale"]) == 0.0,
    "return ranking removed": float(cfg["return_rank_loss_scale"]) == 0.0,
    "action gap retained": cfg["compute_action_gap"] is True,
    "lower delta scale": float(cfg["delta_loss_scale"]) == 0.005,
    "gradient balance enabled": cfg["grad_balance_enabled"] is True,
    "aux/base ratio cap": float(cfg["grad_balance_max_ratio"]) == 0.10,
    "aux absolute cap": float(cfg["grad_balance_max_norm"]) == 25.0,
    "non-finite model step guard": cfg["skip_nonfinite_model_step"] is True,
}
for name, passed in checks.items():
    if not passed:
        raise AssertionError(name)
    print(f"[OK] {name}")
print("Experiment 5E stability audit passed.")
