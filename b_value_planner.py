"""B: replay-grounded Double-DQN + bounded, deterministic RSSM lookahead.

No actor, external evidence modules, imaginary TD targets, or intrinsic rewards.
Replay records use *incoming-action* alignment: action[t] produced state[t].
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BConfig:
    actions: int
    feat_dim: int
    hidden: int = 512
    lr: float = 1e-4
    discount: float = 0.997
    grad_clip: float = 100.0
    target_tau: float = 0.005
    plan_depth: int = 1
    plan_weight: float = 0.30
    plan_delta_clip: float = 0.50
    plan_start: int = 50000
    epsilon_start: float = 1.00
    epsilon_final: float = 0.05
    epsilon_decay: int = 200000
    mineclip_intrinsic_enabled: bool = False
    heatmap_intrinsic_enabled: bool = False
    mineclip_intrinsic_weight: float = 1.0
    heatmap_intrinsic_weight: float = 1.0

    def validate(self):
        assert self.actions >= 2 and self.feat_dim > 0
        assert self.hidden > 0 and self.lr > 0
        assert 0 <= self.discount <= 1
        assert 0 < self.target_tau <= 1
        assert self.plan_depth in (1, 2), 'Only depth 1 or 2 is currently implemented'
        assert 0 <= self.plan_weight <= 1 and self.plan_delta_clip >= 0
        assert 0 <= self.epsilon_final <= self.epsilon_start <= 1
        assert self.epsilon_decay > 0


class QNetwork(nn.Module):
    """Output a Q value for each of the A discrete actions in raw reward units."""
    def __init__(self, feature_dim: int, actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, actions),
        )
        # Start near zero to avoid unstable random initial value scales.
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


def valid_pairs(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(B,T-1) mask, excludes the first observation of a new episode."""
    first = batch['is_first'].bool()
    if first.ndim == 3:
        first = first.squeeze(-1)
    mask = ~first[:, 1:]
    if 'is_last' in batch:
        last = batch['is_last'].bool()
        if last.ndim == 3:
            last = last.squeeze(-1)
        mask &= ~last[:, :-1]
    return mask


def real_td_inputs(
    features: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    actions: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return (s, a, r_next, done_next, s_next, valid_count).

    Note: action[t+1] is responsible for transitioning s[t] -> s[t+1].
    `reward[t+1]` is already the reward *from* that action, not reward[t].
    Use is_terminal for bootstrap (not is_last, which can indicate truncation).
    """
    assert features.ndim == 3 and features.shape[1] >= 2
    batch_size, time_steps, feat_dim = features.shape
    action = batch['action']
    assert action.shape == (batch_size, time_steps, actions), (action.shape, batch_size, time_steps, actions)
    assert torch.all(torch.isfinite(action)), 'Nonfinite action'
    mask = valid_pairs(batch)
    selected = action[:, 1:][mask]
    if selected.numel():
        assert torch.allclose(selected.sum(-1), torch.ones_like(selected[..., 0]), atol=1e-4), 'Valid actions are not one-hot'
        assert torch.all((selected == 0) | (selected == 1)), 'Valid actions must be binary one-hot'

    n = int(mask.sum().item())
    s = features[:, :-1][mask]
    s_next = features[:, 1:][mask]
    a = action[:, 1:][mask].argmax(-1).long()
    r = batch['reward'][:, 1:]
    terminal = batch['is_terminal'][:, 1:]
    if r.ndim == 3:
        r = r.squeeze(-1)
    if terminal.ndim == 3:
        terminal = terminal.squeeze(-1)
    r = r[mask].float()
    terminal = terminal[mask].float()
    assert torch.isfinite(r).all(), 'Nonfinite reward'
    return s, a, r, terminal, s_next, n


def reward_for_learning(batch, mask, task_reward, cfg):
    """Return separate task/teacher targets, never mutating the environment reward."""
    shaped = task_reward.clone()
    components = {}
    for channel, enabled, weight in (
        ("intrinsic_mineclip", cfg.mineclip_intrinsic_enabled, cfg.mineclip_intrinsic_weight),
        ("intrinsic_heatmap", cfg.heatmap_intrinsic_enabled, cfg.heatmap_intrinsic_weight),
    ):
        if enabled and channel not in batch:
            raise KeyError(f"Enabled {channel} is absent from Replay; cannot silently train task-only")
        vals = (batch[channel][:, 1:][mask].float() if enabled else torch.zeros_like(task_reward))
        if vals.ndim == 2 and vals.shape[-1] == 1:
            vals = vals.squeeze(-1)
        assert vals.shape == task_reward.shape, (channel, vals.shape, task_reward.shape)
        components[channel] = vals
        shaped = shaped + (weight * vals)
    return shaped, components


class RealValuePlanner(nn.Module):
    """Replay critic and optional world-model one/two-step rescoring.

    The target Q value is trained only from real replay transitions. The world
    model is never used to construct the TD target. Planning is under no_grad,
    so imagined states cannot update Q or RSSM through a planning loss.
    """
    def __init__(self, config: BConfig):
        super().__init__()
        config.validate()
        self.cfg = config
        self.q = QNetwork(config.feat_dim, config.actions, config.hidden)
        self.q_target = deepcopy(self.q)
        self.q_target.requires_grad_(False)
        self.q_target.eval()
        self._optimizer = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.register_buffer('_updates', torch.tensor(0, dtype=torch.long))

    def epsilon(self, global_steps: int) -> float:
        k = min(max(global_steps, 0) / self.cfg.epsilon_decay, 1.0)
        return self.cfg.epsilon_start + k * (self.cfg.epsilon_final - self.cfg.epsilon_start)

    def train_real(self, post: dict, replay: dict, world_model) -> dict:
        # Important: post was inferred from real RGB observations, not imagined states.
        with torch.no_grad():
            features = world_model.dynamics.get_feat(post).float().detach()
            batch = {k: torch.as_tensor(v, device=features.device) for k, v in replay.items()
                     if k in ('action', 'reward', 'is_first', 'is_last', 'is_terminal', 'intrinsic_mineclip', 'intrinsic_heatmap')}
            s, a, r, done, next_s, count = real_td_inputs(features, batch, self.cfg.actions)
            if count == 0:
                return {'b_valid_pairs': 0.0}
            shaped_r, bonuses = reward_for_learning(batch, valid_pairs(batch), r, self.cfg)
            # Double DQN: choose next greedy action with online Q; evaluate with target Q.
            next_action = self.q(next_s).argmax(-1, keepdim=True)
            next_value = self.q_target(next_s).gather(-1, next_action).squeeze(-1)
            target = shaped_r + self.cfg.discount * (1 - done) * next_value

        # Parent agent may have disabled all module gradients via requires_grad_(False).
        # Explicitly enable Q params for this update; all s and TD targets are detached.
        for p in self.q.parameters():
            p.requires_grad_(True)
        estimate = self.q(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(estimate, target)
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self._optimizer.step()
        with torch.no_grad():
            self._updates.add_(1)
            for tp, p in zip(self.q_target.parameters(), self.q.parameters()):
                tp.mul_(1 - self.cfg.target_tau).add_(p, alpha=self.cfg.target_tau)
            # LayerNorm buffers are not used in this network.
            return {
                'b_valid_pairs': float(count),
                'b_q_loss': float(loss.item()),
                'b_q_grad_norm': float(grad_norm.item()),
                'b_q_pred_mean': float(estimate.mean().item()),
                'b_q_target_mean': float(target.mean().item()),
                'b_positive_pairs': float((r > 0).sum().item()),
                'b_real_reward_mean': float(r.mean().item()),
                'b_td_reward_mean': float(shaped_r.mean().item()),
                'b_intrinsic_mineclip_mean': float(bonuses['intrinsic_mineclip'].mean().item()),
                'b_intrinsic_heatmap_mean': float(bonuses['intrinsic_heatmap'].mean().item()),
                'b_terminal_fraction': float(done.mean().item()),
            }

    @torch.no_grad()
    def _expand_once(self, states: dict, wm) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """Input B states; output BA states, [B,A] rewards and continuation probs."""
        a = self.cfg.actions
        b = next(iter(states.values())).shape[0]
        expanded = {key: value.repeat_interleave(a, dim=0) for key, value in states.items()}
        basis = torch.eye(a, device=next(iter(states.values())).device).repeat(b, 1)
        nxt = wm.dynamics.img_step(expanded, basis, sample=False)
        feat = wm.dynamics.get_feat(nxt).float()
        r = wm.heads['reward'](feat).mode().float().reshape(b, a)
        for channel, enabled, weight in (("intrinsic_mineclip", self.cfg.mineclip_intrinsic_enabled, self.cfg.mineclip_intrinsic_weight), ("intrinsic_heatmap", self.cfg.heatmap_intrinsic_enabled, self.cfg.heatmap_intrinsic_weight)):
            if enabled:
                r = r + weight * wm.heads[channel](feat).mode().float().reshape(b, a)
        p_terminal = wm.heads['end'](feat).mean.float().reshape(b, a).clamp(0, 1)
        continuation = 1 - p_terminal
        # Raw-reward head; no MineCLIP shaping. A predicted *future* state's
        # reward corresponds to executing the incoming action.
        return nxt, r, continuation

    @torch.no_grad()
    def plan_values(self, states: dict, wm) -> torch.Tensor:
        """[B,A] world-model candidate returns, optionally 2-step lookahead."""
        a = self.cfg.actions
        next1, r1, cont1 = self._expand_once(states, wm)
        if self.cfg.plan_depth == 1:
            feat1 = wm.dynamics.get_feat(next1).float()
            leaf = self.q_target(feat1).max(-1).values.reshape(-1, a)
            return r1 + self.cfg.discount * cont1 * leaf
        # depth 2: BA first-step states expanded across A second actions
        next2, r2, cont2 = self._expand_once(next1, wm)
        feat2 = wm.dynamics.get_feat(next2).float()
        leaf = self.q_target(feat2).max(-1).values.reshape(-1, a)
        second = (r2 + self.cfg.discount * cont2 * leaf).max(-1).values.reshape(-1, a)
        return r1 + self.cfg.discount * cont1 * second

    @torch.no_grad()
    def choose(self, states: dict, wm, global_steps: int, training: bool):
        features = wm.dynamics.get_feat(states).float()
        raw_scores = self.q(features)
        scores = raw_scores
        use_plan = self.cfg.plan_weight > 0 and global_steps >= self.cfg.plan_start
        metrics = {'b_plan_active': float(use_plan)}
        if use_plan:
            p = self.plan_values(states, wm)
            # Real replay Q remains the anchor; imagined values have bounded influence.
            delta = (p - raw_scores).clamp(-self.cfg.plan_delta_clip, self.cfg.plan_delta_clip)
            scores = raw_scores + self.cfg.plan_weight * delta
            metrics['b_plan_mean_abs_delta'] = float(delta.abs().mean().item())
            metrics['b_plan_changed_action'] = float((scores.argmax(-1) != raw_scores.argmax(-1)).float().mean().item())
        eps = self.epsilon(global_steps) if training else 0.0
        greedy = scores.argmax(-1)
        explore = torch.rand_like(greedy.float()) < eps
        uniform = torch.randint(self.cfg.actions, greedy.shape, device=greedy.device)
        chosen = torch.where(explore, uniform, greedy)
        # Correct epsilon-greedy behavior probability for the executed discrete action.
        probability = eps / self.cfg.actions + (1 - eps) * (chosen == greedy).float()
        action = F.one_hot(chosen, self.cfg.actions).float()
        logprob = torch.log(probability.clamp_min(1e-12))
        metrics.update({'b_epsilon': eps, 'b_action_id_mean': float(chosen.float().mean().item()),
                        'b_q_max_mean': float(raw_scores.max(-1).values.mean().item())})
        return action, logprob, metrics

    def extra_state_dict(self):
        """Optimizer must be explicitly checkpointed: it is not an nn.Module buffer."""
        return {'q_optimizer': self._optimizer.state_dict()}

    def load_extra_state_dict(self, record):
        self._optimizer.load_state_dict(record['q_optimizer'])
