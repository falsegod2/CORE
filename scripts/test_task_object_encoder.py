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
    "competition_entropy_scale": 0.002,
    "diversity_scale": 0.001,
    "feature_diversity_scale": 0.001,
    "global_semantic_align_scale": 0.02,
}
model = networks.TaskRelevantObjectEncoder(
    shapes, encoder_config, object_config
)
obs = {
    "image": torch.rand(2, 3, 64, 64, 3),
    "task_embedding": torch.randn(2, 3, 512, requires_grad=True),
    "mineclip_embedding": torch.randn(2, 3, 512, requires_grad=True),
}
out = model(obs)
assert out.shape == (2, 3, model.outdim), out.shape
aux = model.get_aux_losses()
assert set(aux) == {
    "task_object_competition_entropy",
    "task_object_diversity",
    "task_object_feature_diversity",
    "task_object_global_semantic_align",
}
loss = out.mean() + sum(value.mean() for value in aux.values())
loss.backward()
assert obs["task_embedding"].grad is None
assert obs["mineclip_embedding"].grad is None
assert model._task_proj.weight.grad is not None
assert model._task_to_objects.weight.grad is not None
assert model._base._cnn.layers[0].weight.grad is not None
metrics = model.get_metrics()
for key in (
    "task_object_relevance_entropy",
    "task_object_top1",
    "task_object_candidate_mass",
    "task_object_assignment_prior_top1",
    "task_object_attention_entropy",
    "task_object_attention_overlap",
    "task_object_feature_overlap",
    "task_object_competition_entropy",
    "task_object_usage_entropy",
    "task_object_sinkhorn_row_error",
    "task_object_sinkhorn_col_error",
    "task_object_gate_mean",
    "task_object_delta_ratio",
    "task_object_diversity_loss",
    "task_object_feature_diversity_loss",
    "task_object_global_semantic_cosine",
    "task_object_semantic_valid",
):
    assert key in metrics, key
assert torch.isfinite(out).all()
assert float(metrics["task_object_sinkhorn_row_error"]) < 1e-3
assert float(metrics["task_object_sinkhorn_col_error"]) < 1e-3
print("PASS", out.shape, {k: float(v) for k, v in metrics.items()})
