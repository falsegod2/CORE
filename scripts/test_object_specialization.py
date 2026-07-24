#!/usr/bin/env python3
"""Verify that V2 anti-collapse objectives can break symmetric attention."""
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("wandb", types.SimpleNamespace())
tb_stub = types.ModuleType("torch.utils.tensorboard")
tb_stub.SummaryWriter = object
sys.modules.setdefault("torch.utils.tensorboard", tb_stub)

import networks

shapes = {
    "image": (64, 64, 3),
    "task_embedding": (512,),
    "mineclip_embedding": (512,),
}
encoder_config = {
    "mlp_keys": "^$",
    "cnn_keys": "^image$",
    "act": "SiLU",
    "norm": True,
    "cnn_depth": 8,
    "kernel_size": 4,
    "minres": 4,
    "mlp_layers": 1,
    "mlp_units": 32,
    "symlog_inputs": True,
}
object_config = {
    "task_key": "task_embedding",
    "task_dim": 512,
    "visual_key": "mineclip_embedding",
    "visual_dim": 512,
    "object_dim": 32,
    "hidden": 32,
    "num_objects": 2,
    "candidate_topk": 6,
    "object_iters": 2,
    "relevance_temperature": 0.5,
    "attention_temperature": 0.5,
    "relevance_bias": 0.25,
    "assignment_uniform_mix": 0.5,
    "sinkhorn_iters": 4,
    "residual_scale": 0.03,
    "output_init": 0.001,
    "competition_entropy_scale": 0.01,
    "diversity_scale": 0.005,
    "feature_diversity_scale": 0.002,
    "global_semantic_align_scale": 0.0,
}

torch.manual_seed(0)
model = networks.TaskRelevantObjectEncoder(
    shapes, encoder_config, object_config
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
obs = {
    "image": torch.rand(8, 3, 64, 64, 3),
    "task_embedding": torch.randn(8, 3, 512),
    "mineclip_embedding": torch.randn(8, 3, 512),
}

model(obs)
initial = float(model.get_metrics()["task_object_attention_overlap"])
for _ in range(40):
    model(obs)
    aux = model.get_aux_losses()
    loss = sum(value.mean() for value in aux.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
model(obs)
final = float(model.get_metrics()["task_object_attention_overlap"])
assert final < initial - 0.25, (initial, final)
assert final < 0.7, (initial, final)
print(
    "SPECIALIZATION PASS: "
    f"attention overlap {initial:.4f} -> {final:.4f}"
)
