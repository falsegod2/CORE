import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("wandb", types.SimpleNamespace())
tb_stub = types.ModuleType("torch.utils.tensorboard")
tb_stub.SummaryWriter = object
sys.modules.setdefault("torch.utils.tensorboard", tb_stub)
import networks

shapes = {
    "image": (64, 64, 3),
    "task_embedding": (512,),
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
fusion_config = {
    "task_key": "task_embedding",
    "task_dim": 512,
    "attention_dim": 32,
    "hidden": 32,
    "topk": 4,
    "temperature": 0.2,
    "residual_scale": 0.05,
    "output_init": 0.001,
}
model = networks.TaskConditionedPatchEncoder(
    shapes, encoder_config, fusion_config
)
obs = {
    "image": torch.rand(2, 3, 64, 64, 3),
    "task_embedding": torch.randn(2, 3, 512, requires_grad=True),
}
out = model(obs)
assert out.shape == (2, 3, model.outdim), out.shape
out.mean().backward()
# task embedding must be detached; fusion parameters and RGB must train.
assert obs["task_embedding"].grad is None
assert model._task_proj.weight.grad is not None
assert model._base._cnn.layers[0].weight.grad is not None
metrics = model.get_metrics()
for key in (
    "task_patch_entropy",
    "task_patch_top1",
    "task_patch_topk_mass",
    "task_patch_gate_mean",
    "task_patch_delta_ratio",
):
    assert key in metrics
print("PASS", out.shape, {k: float(v) for k, v in metrics.items()})
