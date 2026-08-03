import pathlib
import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("wandb", types.SimpleNamespace())
_tb = types.ModuleType("torch.utils.tensorboard")
class _DummyWriter:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return lambda *args, **kwargs: None
_tb.SummaryWriter = _DummyWriter
sys.modules.setdefault("torch.utils.tensorboard", _tb)

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models


def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _ObsSpace:
    spaces = {"image": _Space((64, 64, 3))}


def test_world_model_training_integration():
    loader = yaml.YAML(typ="safe")
    configs = loader.load((ROOT / "configs.yaml").read_text())
    merged = {}
    _recursive_update(merged, configs["defaults"])
    _recursive_update(merged, configs["outcomeaware5d"])

    # Keep the exact module interfaces while reducing dimensions for a CPU test.
    merged.update(
        device="cpu",
        precision=32,
        dyn_hidden=16,
        dyn_deter=16,
        dyn_stoch=4,
        dyn_discrete=4,
        units=32,
        num_actions=3,
        model_lr=1e-4,
        compile=False,
    )
    merged["encoder"] = dict(merged["encoder"])
    merged["encoder"].update(cnn_depth=8, mlp_units=32)
    merged["decoder"] = dict(merged["decoder"])
    merged["decoder"].update(cnn_depth=8, mlp_units=32)
    merged["reward_head"] = dict(merged["reward_head"])
    merged["reward_head"].update(layers=2)
    merged["end_head"] = dict(merged["end_head"])
    merged["end_head"].update(layers=2)

    model = models.WorldModel(_ObsSpace(), None, 0, SimpleNamespace(**merged))
    batch, time, action_dim = 2, 20, 3
    actions = F.one_hot(
        torch.randint(0, action_dim, (batch, time)), action_dim
    ).numpy().astype(np.float32)
    data = {
        "image": np.random.randint(
            0, 256, size=(batch, time, 64, 64, 3), dtype=np.uint8
        ),
        "action": actions,
        "is_first": np.zeros((batch, time), dtype=np.float32),
        "is_terminal": np.zeros((batch, time), dtype=np.float32),
        "reward": np.random.randn(batch, time).astype(np.float32) * 0.01,
    }
    data["is_first"][:, 0] = 1.0

    _, _, _, metrics = model._train(data, torch.tensor(400_000.0))
    assert np.isfinite(metrics["model_loss"])
    assert np.isfinite(metrics["outcome_aware_total_loss"])
    assert "oa_delta_cosine_h8" in metrics
    assert "oa_return_pearson_h8" in metrics
