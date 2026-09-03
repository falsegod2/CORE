#!/usr/bin/env python3
"""Small CPU integration smoke test for models_dreamer + full-latent Proto."""

import copy
import sys
import types
import numpy as np
import torch
from ruamel.yaml import YAML

# The container used for this static/integration check does not install wandb.
# Production training requirements already list it; a dummy module is enough
# because this smoke test never constructs the logger.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
_tb = types.ModuleType("torch.utils.tensorboard")
class _DummySummaryWriter:
    def __init__(self, *args, **kwargs): pass
    def add_scalar(self, *args, **kwargs): pass
    def add_image(self, *args, **kwargs): pass
    def add_video(self, *args, **kwargs): pass
    def flush(self): pass
_tb.SummaryWriter = _DummySummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", _tb)

import ablation
import models_dreamer


class Space:
    def __init__(self, shape):
        self.shape = tuple(shape)


class ObsSpace:
    def __init__(self):
        self.spaces = {
            "image": Space((64, 64, 3)),
            "heatmap": Space((64, 64, 1)),
            "is_first": Space((1,)),
            "is_last": Space((1,)),
            "is_terminal": Space((1,)),
            "intrinsic": Space((1,)),
            "task_score": Space((1,)),
        }


def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def load_config():
    yaml = YAML(typ="safe", pure=True)
    cfgs = yaml.load(open("configs.yaml", "r"))
    merged = {}
    recursive_update(merged, cfgs["defaults"])
    recursive_update(merged, cfgs["minedojo"])
    recursive_update(merged, cfgs["ab_dreamer_proto"])

    # Tiny network for a CPU smoke test only.
    merged.update({
        "device": "cpu",
        "precision": 32,
        "compile": False,
        "dyn_stoch": 4,
        "dyn_deter": 32,
        "dyn_hidden": 32,
        "dyn_discrete": 4,
        "units": 64,
        "num_actions": 5,
        "batch_size": 2,
        "batch_length": 20,
        "model_lr": 1e-4,
    })
    merged["encoder"] = dict(merged["encoder"])
    merged["encoder"]["cnn_depth"] = 8
    merged["decoder"] = dict(merged["decoder"])
    merged["decoder"]["cnn_depth"] = 8
    merged["reward_head"] = dict(merged["reward_head"])
    merged["reward_head"]["layers"] = 2
    merged["end_head"] = dict(merged["end_head"])
    merged["end_head"]["layers"] = 2
    merged["intrinsic_head"] = dict(merged["intrinsic_head"])
    merged["intrinsic_head"]["layers"] = 2
    merged["s_multi_step_consistency"] = copy.deepcopy(merged["s_multi_step_consistency"])
    merged["s_multi_step_consistency"]["starts_per_sequence"] = 2
    cfg = types.SimpleNamespace(**merged)
    return ablation.apply_ablation_config(cfg)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = load_config()
    assert cfg.model_backend == "dreamer"
    assert cfg.full_latent_proto_enabled
    assert not cfg.s_multi_step_consistency["enabled"]
    assert cfg.decoder["cnn_keys"] == "^image$"

    wm = models_dreamer.WorldModel(ObsSpace(), None, None, cfg)
    b, t, a = 2, 20, cfg.num_actions
    action_idx = np.random.randint(0, a, size=(b, t))
    action = np.eye(a, dtype=np.float32)[action_idx]
    image = np.random.randint(0, 256, size=(b, t, 64, 64, 3), dtype=np.uint8)
    heatmap = np.random.randint(0, 256, size=(b, t, 64, 64), dtype=np.uint8)
    task_score = np.linspace(0.02, 0.98, t, dtype=np.float32)[None, :, None]
    task_score = np.repeat(task_score, b, axis=0)
    reward = np.zeros((b, t), dtype=np.float32)
    reward[0, 18] = 1.0
    intrinsic = np.zeros((b, t), dtype=np.float32)
    intrinsic[:, 6] = 0.2
    is_first = np.zeros((b, t), dtype=np.float32)
    is_first[:, 0] = 1.0
    is_terminal = np.zeros((b, t), dtype=np.float32)

    data = {
        "image": image,
        "heatmap": heatmap,
        "task_score": task_score,
        "intrinsic": intrinsic,
        "reward": reward,
        "action": action,
        "is_first": is_first,
        "is_terminal": is_terminal,
    }
    _, _, _, metrics = wm._train(data)
    assert metrics["proto_latent_source_full"] == 1.0
    assert np.isfinite(metrics["proto_weighted_loss"])
    assert "proto_acc_h15" in metrics
    assert "s_multistep_loss" not in metrics
    print("PASS: single-stream Dreamer world model trains with Generic Proto only.")
    print("model_loss=", float(metrics["model_loss"]))
    print("proto_weighted_loss=", float(metrics["proto_weighted_loss"]))
    print("proto_acc_h15=", float(metrics["proto_acc_h15"]))


if __name__ == "__main__":
    main()
