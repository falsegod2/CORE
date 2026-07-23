#!/usr/bin/env python3
from pathlib import Path
import sys
import types
import torch

# The unit test exercises only the encoder. Stub optional logging packages that
# are imported transitively by tools.py in minimal build environments.
sys.modules.setdefault("wandb", types.SimpleNamespace())
tb_stub = types.ModuleType("torch.utils.tensorboard")
tb_stub.SummaryWriter = object
sys.modules.setdefault("torch.utils.tensorboard", tb_stub)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import networks

shapes = {
    "image": (64, 64, 3),
    "mineclip_embedding": (512,),
    "mineclip_reward": (1,),
    "is_first": (1,),
    "is_last": (1,),
    "is_terminal": (1,),
}
encoder_cfg = dict(
    mlp_keys="inventory|inventory_max|equipped|health|hunger|breath|obs_reward",
    cnn_keys="^image$",
    act="SiLU", norm=True, cnn_depth=8, kernel_size=4, minres=4,
    mlp_layers=2, mlp_units=32, symlog_inputs=True,
)
fusion_cfg = dict(
    key="mineclip_embedding", input_dim=512, fusion_dim=32, hidden=32,
    normalize_input=True, gate_bias=-2.0, residual_scale=0.1,
    output_init=1e-3,
)
encoder = networks.GatedMineCLIPEncoder(shapes, encoder_cfg, fusion_cfg)
image = torch.rand(2, 3, 64, 64, 3)
semantic = torch.randn(2, 3, 512, requires_grad=True)
obs = {"image": image, "mineclip_embedding": semantic}
out = encoder(obs)
assert out.shape[:2] == (2, 3)
assert out.shape[-1] == encoder.outdim
loss = out.square().mean()
loss.backward()
assert semantic.grad is None, "MineCLIP observation must be stop-gradient"
assert encoder._fusion_out.weight.grad is not None
metrics = encoder.get_metrics()
for key in ("mineclip_gate_mean", "mineclip_gate_std", "mineclip_delta_ratio"):
    assert key in metrics
print(f"TEST PASSED: output={tuple(out.shape)}, gate_mean={float(metrics['mineclip_gate_mean']):.4f}")
