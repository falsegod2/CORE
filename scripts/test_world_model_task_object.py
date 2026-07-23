#!/usr/bin/env python3
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("wandb", types.SimpleNamespace())
tb_stub = types.ModuleType("torch.utils.tensorboard")
tb_stub.SummaryWriter = object
sys.modules.setdefault("torch.utils.tensorboard", tb_stub)

import models


class Space:
    def __init__(self, shape):
        self.shape = shape


class ObsSpace:
    def __init__(self):
        self.spaces = {
            key: Space(shape)
            for key, shape in {
                "image": (64, 64, 3),
                "task_embedding": (512,),
                "mineclip_embedding": (512,),
                "is_first": (1,),
                "is_last": (1,),
                "is_terminal": (1,),
                "mineclip_reward": (1,),
            }.items()
        }


with (ROOT / "configs.yaml").open() as file:
    config = yaml.safe_load(file)["defaults"]
config.update(
    device="cpu",
    precision=32,
    num_actions=4,
    dyn_hidden=32,
    dyn_deter=64,
    dyn_stoch=4,
    dyn_discrete=4,
    units=64,
    model_lr=1e-4,
    grad_clip=100.0,
    opt_eps=1e-8,
    weight_decay=0.0,
)
config["encoder"] = dict(
    config["encoder"], cnn_depth=8, mlp_layers=2, mlp_units=32
)
config["decoder"] = dict(
    config["decoder"], cnn_depth=8, mlp_layers=2, mlp_units=32
)
config["task_object_tokens"] = dict(
    config["task_object_tokens"], object_dim=32, hidden=32
)
for key in ("reward_head", "end_head", "mineclip_head"):
    config[key] = dict(config[key], layers=2)
config = SimpleNamespace(**config)

world_model = models.WorldModel(ObsSpace(), None, 0, config)
batch, time = 2, 5
actions = np.eye(4, dtype=np.float32)[
    np.random.randint(0, 4, size=(batch, time))
]
data = {
    "image": np.random.randint(
        0, 256, (batch, time, 64, 64, 3), dtype=np.uint8
    ),
    "task_embedding": np.random.randn(batch, time, 512).astype(np.float16),
    "mineclip_embedding": np.random.randn(batch, time, 512).astype(np.float16),
    "mineclip_reward": (
        0.01 * np.random.randn(batch, time, 1)
    ).astype(np.float32),
    "action": actions,
    "is_first": np.zeros((batch, time), dtype=np.float32),
    "is_terminal": np.zeros((batch, time), dtype=np.float32),
    "reward": np.random.randn(batch, time, 1).astype(np.float32),
}
data["is_first"][:, 0] = 1.0
_, _, context, metrics = world_model._train(data)
assert context["embed"].shape[:2] == (batch, time)
for key in (
    "task_object_relevance_entropy",
    "task_object_candidate_mass",
    "task_object_attention_overlap",
    "task_object_delta_ratio",
    "task_object_coverage_scaled_loss",
    "task_object_diversity_scaled_loss",
    "task_object_semantic_align_scaled_loss",
    "task_object_semantic_cosine",
):
    assert key in metrics, key
print(
    "TEST PASSED: world-model update; "
    f"embed={tuple(context['embed'].shape)}, "
    f"candidate_mass={float(metrics['task_object_candidate_mass']):.4f}, "
    f"overlap={float(metrics['task_object_attention_overlap']):.4f}"
)
