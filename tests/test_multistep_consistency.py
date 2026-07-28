import pathlib
import sys
import types

# The unit test only needs RSSM math; stub optional logging dependency.
sys.modules.setdefault("wandb", types.SimpleNamespace())
_tb = types.ModuleType("torch.utils.tensorboard")
class _DummyWriter:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return lambda *args, **kwargs: None
_tb.SummaryWriter = _DummyWriter
sys.modules.setdefault("torch.utils.tensorboard", _tb)

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networks
from multistep_consistency import MultiStepRSSMConsistency


def test_loss_is_finite_and_backpropagates():
    torch.manual_seed(7)
    batch, time, action_dim, embed_dim = 2, 20, 3, 10
    rssm = networks.RSSM(
        stoch=4,
        deter=8,
        hidden=16,
        rec_depth=1,
        discrete=4,
        num_actions=action_dim,
        embed=embed_dim,
        device="cpu",
    )
    embeds = torch.randn(batch, time, embed_dim)
    action_ids = torch.randint(0, action_dim, (batch, time))
    actions = F.one_hot(action_ids, action_dim).float()
    is_first = torch.zeros(batch, time)
    is_first[:, 0] = 1.0
    post, _ = rssm.observe(embeds, actions, is_first)

    objective = MultiStepRSSMConsistency(
        feat_dim=4 * 4 + 8,
        horizons=(1, 2, 4, 8),
        horizon_weights=(1.0, 1.0, 0.75, 0.5),
        projection_dim=12,
        starts_per_sequence=2,
    )
    loss, metrics = objective(rssm, post, actions, is_first)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "multistep_loss_h8" in metrics
    loss.backward()
    grad_sum = sum(
        float(p.grad.abs().sum())
        for p in rssm.parameters()
        if p.grad is not None
    )
    assert grad_sum > 0.0


def test_masks_episode_boundaries():
    torch.manual_seed(11)
    batch, time, action_dim, embed_dim = 1, 20, 3, 10
    rssm = networks.RSSM(
        stoch=4, deter=8, hidden=16, discrete=4,
        num_actions=action_dim, embed=embed_dim, device="cpu"
    )
    embeds = torch.randn(batch, time, embed_dim)
    actions = F.one_hot(torch.randint(0, action_dim, (batch, time)), action_dim).float()
    is_first = torch.zeros(batch, time)
    is_first[:, 0] = 1.0
    is_first[:, 10] = 1.0
    post, _ = rssm.observe(embeds, actions, is_first)
    objective = MultiStepRSSMConsistency(
        feat_dim=24, horizons=(1, 4, 8),
        horizon_weights=(1.0, 1.0, 1.0),
        projection_dim=12, starts_per_sequence=3,
    )
    loss, metrics = objective(rssm, post, actions, is_first)
    assert torch.isfinite(loss)
    assert metrics["multistep_valid_h8"] < 1.0
