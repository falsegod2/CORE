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

import torch
from torch import nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networks
from gradient_balance import balance_auxiliary_loss
from outcome_aware_multistep import OutcomeAwareMultiStepRSSM


class _ScalarDist:
    def __init__(self, value):
        self._value = value
    def mode(self):
        return self._value


class _RewardHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.linear = nn.Linear(feat_dim, 1)
    def forward(self, feat):
        return _ScalarDist(self.linear(feat))


def _batch(batch=2, time=16, action_dim=3, embed_dim=10):
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
    rewards = torch.zeros(batch, time)
    ends = torch.zeros(batch, time)
    return rssm, post, actions, is_first, rewards, ends


def _stable_objective(**overrides):
    kwargs = dict(
        feat_dim=24,
        horizons=(1, 2, 4, 8),
        horizon_weights=(0.25, 0.5, 0.75, 1.0),
        projection_dim=12,
        starts_per_sequence=2,
        absolute_loss_scale=0.0025,
        delta_loss_scale=0.005,
        return_loss_scale=0.005,
        return_rank_loss_scale=0.0,
        action_margin_loss_scale=0.0,
        compute_action_gap=True,
        rollout_sample=True,
        preserve_rng_state=True,
        detach_start_state=True,
        global_start_step=0,
        global_ramp_steps=0,
        horizon_start_steps=(0, 0, 0, 0),
        horizon_ramp_steps=(0, 0, 0, 0),
    )
    kwargs.update(overrides)
    return OutcomeAwareMultiStepRSSM(**kwargs)


def test_stochastic_auxiliary_rollout_preserves_main_rng_stream():
    torch.manual_seed(1234)
    rssm, post, actions, is_first, rewards, ends = _batch()
    reward_head = _RewardHead(24)
    objective = _stable_objective()
    before = torch.random.get_rng_state().clone()
    loss, metrics = objective(
        rssm, reward_head, post, actions, is_first, rewards, ends, env_step=500_000
    )
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert torch.isfinite(loss)
    assert "oa_action_gap_h8" in metrics


def test_disabled_rank_and_margin_are_exact_zero_with_constant_returns():
    torch.manual_seed(44)
    rssm, post, actions, is_first, rewards, ends = _batch()
    reward_head = _RewardHead(24)
    objective = _stable_objective()
    loss, metrics = objective(
        rssm, reward_head, post, actions, is_first, rewards, ends, env_step=500_000
    )
    assert torch.isfinite(loss)
    assert torch.allclose(metrics["oa_rank_contribution"], torch.zeros_like(loss))
    assert torch.allclose(metrics["oa_action_contribution"], torch.zeros_like(loss))
    assert torch.isfinite(metrics["oa_rank_accuracy_h8"])


def test_auxiliary_gradient_balancing_caps_ratio_on_shared_support():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    base_loss = parameter.square().sum()
    auxiliary_loss = 1000.0 * parameter.square().sum()
    balanced, metrics = balance_auxiliary_loss(
        base_loss,
        auxiliary_loss,
        [parameter],
        max_ratio=0.10,
        max_norm=25.0,
    )
    assert torch.isfinite(balanced)
    assert float(metrics["oa_aux_grad_scale"]) < 1.0

    base_grad = torch.autograd.grad(base_loss, parameter, retain_graph=True)[0]
    aux_grad = torch.autograd.grad(balanced, parameter, retain_graph=True)[0]
    ratio = aux_grad.norm() / base_grad.norm()
    assert float(ratio) <= 0.10001


def test_stable_profile_has_no_h15_training_path():
    objective = _stable_objective()
    assert objective.horizons == (1, 2, 4, 8)
    assert objective.max_horizon == 8
    assert objective.return_rank_loss_scale == 0.0
    assert objective.action_margin_loss_scale == 0.0


def test_world_model_5e_profile_integration():
    import numpy as np
    import ruamel.yaml as yaml
    from types import SimpleNamespace
    import models

    def merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    class Space:
        def __init__(self, shape): self.shape = shape
    class ObsSpace:
        spaces = {"image": Space((64, 64, 3))}

    loader = yaml.YAML(typ="safe")
    configs = loader.load((ROOT / "configs.yaml").read_text())
    merged = {}
    merge(merged, configs["defaults"])
    merge(merged, configs["outcomeaware5e"])
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

    model = models.WorldModel(ObsSpace(), None, 0, SimpleNamespace(**merged))
    batch, time, action_dim = 2, 16, 3
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
    _, _, _, metrics = model._train(data, torch.tensor(500_000.0))
    assert np.isfinite(metrics["model_loss"])
    assert np.isfinite(metrics["oa_aux_grad_norm"])
    assert 0.0 <= metrics["oa_aux_grad_scale"] <= 1.0
    assert "oa_action_gap_h8" in metrics
    assert "oa_abs_cosine_h15" not in metrics


def test_optimizer_skips_nonfinite_gradient_step():
    import tools
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = tools.Optimizer(
        "guard",
        [parameter],
        lr=1e-3,
        eps=1e-4,
        clip=1000,
        wd=0.0,
        use_amp=False,
        skip_nonfinite=True,
    )
    before = parameter.detach().clone()
    loss = parameter.sum() * torch.tensor(float("inf"))
    metrics = optimizer(loss, [parameter], retain_graph=False)
    assert metrics["guard_step_skipped_nonfinite"] == 1.0
    assert torch.equal(parameter.detach(), before)
