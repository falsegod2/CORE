import pathlib
import sys
import types

# Optional logging packages are irrelevant to the unit tests.
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
from outcome_aware_multistep import OutcomeAwareMultiStepRSSM


class _ScalarDist:
    def __init__(self, value):
        self._value = value
    def mode(self):
        return self._value


class LinearRewardHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.linear = nn.Linear(feat_dim, 1)
    def forward(self, feat):
        return _ScalarDist(self.linear(feat))


def _make_real_rssm_batch(batch=2, time=20, action_dim=3, embed_dim=10):
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
    embeds = torch.randn(batch, time, embed_dim, requires_grad=True)
    action_ids = torch.randint(0, action_dim, (batch, time))
    actions = F.one_hot(action_ids, action_dim).float()
    is_first = torch.zeros(batch, time)
    is_first[:, 0] = 1.0
    post, _ = rssm.observe(embeds, actions, is_first)
    rewards = torch.randn(batch, time) * 0.05
    ends = torch.zeros(batch, time)
    return rssm, embeds, post, actions, is_first, rewards, ends


def _objective(**overrides):
    kwargs = dict(
        feat_dim=4 * 4 + 8,
        horizons=(1, 2, 4, 8),
        horizon_weights=(0.25, 0.5, 0.75, 1.0),
        projection_dim=12,
        starts_per_sequence=2,
        absolute_loss_scale=0.005,
        delta_loss_scale=0.020,
        return_loss_scale=0.010,
        return_rank_loss_scale=0.005,
        action_margin_loss_scale=0.005,
        global_start_step=0,
        global_ramp_steps=0,
        horizon_start_steps=(0, 0, 0, 0),
        horizon_ramp_steps=(0, 0, 0, 0),
        rollout_sample=False,
    )
    kwargs.update(overrides)
    return OutcomeAwareMultiStepRSSM(**kwargs)


def test_full_objective_is_finite_and_updates_rssm_and_reward_head():
    torch.manual_seed(7)
    rssm, embeds, post, actions, is_first, rewards, ends = _make_real_rssm_batch()
    reward_head = LinearRewardHead(24)
    objective = _objective()

    loss, metrics = objective(
        rssm, reward_head, post, actions, is_first, rewards, ends, env_step=400_000
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "oa_delta_cosine_h8" in metrics
    assert "oa_return_pearson_h8" in metrics
    assert "oa_action_gap_h8" in metrics
    loss.backward()

    rssm_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in rssm.parameters()
        if parameter.grad is not None
    )
    reward_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in reward_head.parameters()
        if parameter.grad is not None
    )
    assert rssm_grad > 0.0
    assert reward_grad > 0.0
    # Start and target posterior states are detached/no-grad targets. The
    # auxiliary objective must not update the observation encoder path.
    assert embeds.grad is None


def test_curriculum_is_zero_before_warmup_and_active_after_ramp():
    torch.manual_seed(9)
    rssm, _, post, actions, is_first, rewards, ends = _make_real_rssm_batch()
    reward_head = LinearRewardHead(24)
    objective = _objective(
        global_start_step=50_000,
        global_ramp_steps=250_000,
        horizon_start_steps=(50_000, 50_000, 50_000, 150_000),
        horizon_ramp_steps=(100_000, 100_000, 100_000, 100_000),
    )
    early, early_metrics = objective(
        rssm, reward_head, post, actions, is_first, rewards, ends, env_step=0
    )
    late, late_metrics = objective(
        rssm, reward_head, post, actions, is_first, rewards, ends, env_step=400_000
    )
    assert torch.allclose(early, torch.zeros_like(early))
    assert float(early_metrics["outcome_aware_curriculum_scale"]) == 0.0
    assert float(late_metrics["outcome_aware_curriculum_scale"]) == 1.0
    assert torch.isfinite(late) and float(late.detach()) >= 0.0


def test_episode_boundaries_reduce_long_horizon_validity():
    torch.manual_seed(11)
    rssm, _, post, actions, is_first, rewards, ends = _make_real_rssm_batch(
        batch=1, time=20
    )
    is_first[:, 10] = 1.0
    # Recompute post so the test trajectory itself respects the reset.
    embeds = torch.randn(1, 20, 10)
    post, _ = rssm.observe(embeds, actions, is_first)
    objective = _objective(starts_per_sequence=3)
    reward_head = LinearRewardHead(24)
    loss, metrics = objective(
        rssm, reward_head, post, actions, is_first, rewards, ends, env_step=400_000
    )
    assert torch.isfinite(loss)
    assert metrics["oa_valid_h8"] < metrics["oa_valid_h1"]


class FakeDynamics(nn.Module):
    _discrete = False
    def img_step(self, prev_state, prev_action, sample=False):
        increment = prev_action[..., :1]
        deter = prev_state["deter"] + increment
        mean = prev_state["mean"] + increment
        return {
            "deter": deter,
            "mean": mean,
            "std": torch.ones_like(mean),
            "stoch": mean,
        }
    def get_feat(self, state):
        return torch.cat([state["stoch"], state["deter"]], dim=-1)


class ZeroRewardHead(nn.Module):
    def forward(self, feat):
        return _ScalarDist(feat[..., :1] * 0.0)


def _fake_posterior(batch, time):
    base = torch.arange(time, dtype=torch.float32)[None, :, None].repeat(batch, 1, 1)
    return {
        "deter": base,
        "mean": base,
        "std": torch.ones_like(base),
        "stoch": base,
    }


def test_prefix_return_target_alignment_and_terminal_stop():
    dynamics = FakeDynamics()
    reward_head = ZeroRewardHead()
    actions = torch.ones(1, 5, 1)
    is_first = torch.zeros(1, 5)
    is_first[:, 0] = 1.0
    rewards = torch.tensor([[0.0, 1.0, 2.0, 99.0, 99.0]])
    ends = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    objective = OutcomeAwareMultiStepRSSM(
        feat_dim=2,
        horizons=(1, 2),
        horizon_weights=(1.0, 1.0),
        projection_dim=2,
        starts_per_sequence=1,
        discount=0.5,
        absolute_loss_scale=0.0,
        delta_loss_scale=0.0,
        return_loss_scale=1.0,
        return_rank_loss_scale=0.0,
        action_margin_loss_scale=0.0,
        compute_action_gap=False,
        global_start_step=0,
        global_ramp_steps=0,
        horizon_start_steps=(0, 0),
        horizon_ramp_steps=(0, 0),
    )
    loss, metrics = objective(
        dynamics,
        reward_head,
        _fake_posterior(1, 5),
        actions,
        is_first,
        rewards,
        ends,
        env_step=1,
    )
    assert torch.isfinite(loss)
    # action/reward index t+1 is the first future transition.
    assert torch.allclose(metrics["oa_return_target_mean_h1"], torch.tensor(1.0))
    # terminal reward at step 2 is retained: 1 + 0.5 * 2 = 2.
    assert torch.allclose(metrics["oa_return_target_mean_h2"], torch.tensor(2.0))


def test_deterministic_auxiliary_rollout_does_not_consume_global_rng():
    torch.manual_seed(123)
    rssm, _, post, actions, is_first, rewards, ends = _make_real_rssm_batch()
    reward_head = LinearRewardHead(24)
    objective = _objective(rollout_sample=False)
    before = torch.random.get_rng_state().clone()
    objective(rssm, reward_head, post, actions, is_first, rewards, ends, 400_000)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_action_gap_uses_real_action_against_shuffled_counterfactual():
    dynamics = FakeDynamics()
    reward_head = ZeroRewardHead()
    batch, time = 2, 3
    posterior = {
        "deter": torch.tensor([[[0.0], [1.0], [1.0]], [[0.0], [-1.0], [-1.0]]]),
        "mean": torch.tensor([[[0.0], [1.0], [1.0]], [[0.0], [-1.0], [-1.0]]]),
        "std": torch.ones(batch, time, 1),
        "stoch": torch.tensor([[[0.0], [1.0], [1.0]], [[0.0], [-1.0], [-1.0]]]),
    }
    actions = torch.zeros(batch, time, 1)
    actions[0, 1, 0] = 1.0
    actions[1, 1, 0] = -1.0
    is_first = torch.zeros(batch, time)
    is_first[:, 0] = 1.0
    rewards = torch.zeros(batch, time)
    ends = torch.zeros(batch, time)
    objective = OutcomeAwareMultiStepRSSM(
        feat_dim=2,
        horizons=(1,),
        horizon_weights=(1.0,),
        projection_dim=2,
        starts_per_sequence=1,
        absolute_loss_scale=0.0,
        delta_loss_scale=0.0,
        return_loss_scale=0.0,
        return_rank_loss_scale=0.0,
        action_margin_loss_scale=1.0,
        action_margin=0.1,
        compute_action_gap=True,
        global_start_step=0,
        global_ramp_steps=0,
        horizon_start_steps=(0,),
        horizon_ramp_steps=(0,),
    )
    loss, metrics = objective(
        dynamics, reward_head, posterior, actions, is_first, rewards, ends, 1
    )
    assert torch.isfinite(loss)
    assert float(metrics["oa_action_gap_h1"]) > 1.9
    assert torch.allclose(metrics["oa_action_margin_loss_h1"], torch.tensor(0.0))
