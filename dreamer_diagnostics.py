"""Detached internal-state diagnostics for the clean DreamerV3 baseline.

These metrics are observation-only.  They never contribute to any loss and
never update Dreamer parameters.  Deterministic RSSM mode rollouts are used so
logging does not consume PyTorch's global RNG state.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TensorDict = Mapping[str, torch.Tensor]


class FixedRandomProjector(nn.Module):
    """Frozen projection matching Stable's diagnostic geometry.

    A private CPU generator avoids perturbing the global RNG sequence.
    """

    def __init__(self, input_dim: int, output_dim: int = 512, seed: int = 314159):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weight = torch.randn(int(output_dim), int(input_dim), generator=generator)
        weight = weight / math.sqrt(float(output_dim))
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight)


class DreamerDiagnostics(nn.Module):
    """Counterfactual, return-quality, and task-separability diagnostics."""

    def __init__(
        self,
        feat_dim: int,
        horizons: Sequence[int] = (1, 2, 4, 8, 15),
        starts_per_sequence: int = 4,
        projection_dim: int = 512,
        projection_seed: int = 314159,
        discount: float = 0.997,
        task_probe: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        horizons = tuple(int(x) for x in horizons)
        if not horizons or any(x <= 0 for x in horizons):
            raise ValueError(f"Invalid diagnostic horizons: {horizons}")
        if sorted(set(horizons)) != list(horizons):
            raise ValueError(f"Diagnostic horizons must be sorted/unique: {horizons}")
        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.starts_per_sequence = int(starts_per_sequence)
        self.discount = float(discount)
        self.task_probe = bool(task_probe)
        self.eps = float(eps)
        self.projector = FixedRandomProjector(
            int(feat_dim), int(projection_dim), int(projection_seed)
        )

    @staticmethod
    def _as_tensor(value, device, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _detach_state(state: TensorDict) -> Dict[str, torch.Tensor]:
        return {key: value.detach() for key, value in state.items()}

    @staticmethod
    def _gather_state(state: TensorDict, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {key: value[:, indices] for key, value in state.items()}

    @staticmethod
    def _flatten_batch_starts(state: TensorDict) -> Dict[str, torch.Tensor]:
        return {
            key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            for key, value in state.items()
        }

    def _start_indices(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        usable = int(sequence_length) - self.max_horizon
        if usable <= 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        count = min(self.starts_per_sequence, usable)
        if count <= 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(0, usable - 1, steps=count, device=device).round().long()

    @staticmethod
    def _smooth_feat(dynamics: nn.Module, state: TensorDict) -> torch.Tensor:
        """Use distribution probabilities + deterministic state, not sampled one-hot."""
        deter = state["deter"]
        if getattr(dynamics, "_discrete", False):
            probs = torch.softmax(state["logit"], dim=-1)
            probs = probs.reshape(*probs.shape[:-2], -1)
            stoch = probs
        else:
            stoch = state["mean"]
        return torch.cat([stoch, deter], dim=-1)

    def _project_state(self, dynamics: nn.Module, state: TensorDict) -> torch.Tensor:
        feat = self._smooth_feat(dynamics, state).float()
        return F.normalize(self.projector(feat), dim=-1, eps=self.eps)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_corr(
        x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        mask = mask.to(x.dtype)
        denom = mask.sum().clamp_min(1.0)
        mx = (x * mask).sum() / denom
        my = (y * mask).sum() / denom
        xc = (x - mx) * mask
        yc = (y - my) * mask
        cov = (xc * yc).sum() / denom
        vx = xc.square().sum() / denom
        vy = yc.square().sum() / denom
        corr = cov / torch.sqrt((vx * vy).clamp_min(eps))
        # Correlation is undefined for constant vectors; report 0 instead of noise.
        good = (vx > eps) & (vy > eps) & (mask.sum() >= 2)
        return torch.where(good, corr, torch.zeros_like(corr))

    def _mode_rollout(
        self,
        dynamics: nn.Module,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        current = self._detach_state(start_state)
        captured: Dict[int, Dict[str, torch.Tensor]] = {}
        wanted = set(self.horizons)
        for step in range(1, self.max_horizon + 1):
            current = dynamics.img_step(
                current, action_prefix[:, step - 1], sample=False
            )
            if step in wanted:
                captured[step] = self._detach_state(current)
        return captured

    def _counterfactual_metrics(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        starts: torch.Tensor,
        first_prefix: torch.Tensor,
        batch: int,
        num_starts: int,
    ):
        true_states = self._mode_rollout(dynamics, start_state, action_prefix.detach())
        if action_prefix.shape[0] > 1:
            shuffled_actions = torch.roll(action_prefix.detach(), shifts=1, dims=0)
        else:
            shuffled_actions = torch.flip(action_prefix.detach(), dims=[1])
        zero_actions = torch.zeros_like(action_prefix)
        shuffled_states = self._mode_rollout(dynamics, start_state, shuffled_actions)
        zero_states = self._mode_rollout(dynamics, start_state, zero_actions)

        metrics = {}
        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=action_prefix.device
        )
        for step in range(1, self.max_horizon + 1):
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])
            if step not in true_states:
                continue
            target = self._flatten_batch_starts(
                self._gather_state(posterior, starts + step)
            )
            target_proj = self._project_state(dynamics, target)
            true_proj = self._project_state(dynamics, true_states[step])
            shuffle_proj = self._project_state(dynamics, shuffled_states[step])
            zero_proj = self._project_state(dynamics, zero_states[step])

            true_target = (true_proj * target_proj).sum(-1)
            shuffle_target = (shuffle_proj * target_proj).sum(-1)
            zero_target = (zero_proj * target_proj).sum(-1)
            true_shuffle = (true_proj * shuffle_proj).sum(-1)
            true_zero = (true_proj * zero_proj).sum(-1)
            mask = valid_prefix.reshape(-1)

            t = self._masked_mean(true_target, mask)
            s = self._masked_mean(shuffle_target, mask)
            z = self._masked_mean(zero_target, mask)
            metrics[f"dreamer_multistep_cosine_h{step}"] = t
            metrics[f"dreamer_cf_shuffle_target_cos_h{step}"] = s
            metrics[f"dreamer_cf_zero_target_cos_h{step}"] = z
            metrics[f"dreamer_cf_margin_shuffle_h{step}"] = t - s
            metrics[f"dreamer_cf_margin_zero_h{step}"] = t - z
            metrics[f"dreamer_cf_response_gap_shuffle_h{step}"] = 1.0 - self._masked_mean(
                true_shuffle, mask
            )
            metrics[f"dreamer_cf_response_gap_zero_h{step}"] = 1.0 - self._masked_mean(
                true_zero, mask
            )
            metrics[f"dreamer_cf_valid_h{step}"] = mask.float().mean()
        return metrics, true_states

    def _return_metrics(
        self,
        dynamics: nn.Module,
        reward_head: nn.Module,
        posterior: TensorDict,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        starts: torch.Tensor,
        first_prefix: torch.Tensor,
        rewards: torch.Tensor,
        batch: int,
        num_starts: int,
    ):
        metrics = {}
        current = self._detach_state(start_state)
        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=action_prefix.device
        )
        pred_return = torch.zeros(
            batch, num_starts, device=action_prefix.device, dtype=torch.float32
        )
        true_return = torch.zeros_like(pred_return)

        for step in range(1, self.max_horizon + 1):
            current = dynamics.img_step(
                current, action_prefix[:, step - 1], sample=False
            )
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])

            feat = dynamics.get_feat(current)
            pred_reward = reward_head(feat).mode()
            if pred_reward.ndim > 1 and pred_reward.shape[-1] == 1:
                pred_reward = pred_reward[..., 0]
            pred_reward = pred_reward.reshape(batch, num_starts).float()
            true_reward = rewards[:, starts + step].float()
            pred_return = pred_return + (self.discount ** (step - 1)) * pred_reward
            true_return = true_return + (self.discount ** (step - 1)) * true_reward

            if step not in self.horizons:
                continue
            mask = valid_prefix.reshape(-1)
            pred_flat = pred_return.reshape(-1)
            true_flat = true_return.reshape(-1)
            metrics[f"dreamer_return_corr_h{step}"] = self._masked_corr(
                pred_flat, true_flat, mask, self.eps
            )
            metrics[f"dreamer_return_mae_h{step}"] = self._masked_mean(
                (pred_flat - true_flat).abs(), mask
            )
            metrics[f"dreamer_return_pred_std_h{step}"] = self._masked_std(
                pred_flat, mask
            )
            metrics[f"dreamer_return_target_std_h{step}"] = self._masked_std(
                true_flat, mask
            )
            metrics[f"dreamer_return_valid_h{step}"] = mask.float().mean()
        return metrics

    def _masked_std(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        mean = (values * mask_f).sum() / denom
        var = ((values - mean).square() * mask_f).sum() / denom
        return torch.sqrt(var.clamp_min(0.0))

    def _reanchor_metrics(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        starts: torch.Tensor,
        first_prefix: torch.Tensor,
        batch: int,
        num_starts: int,
        true_states: Dict[int, Dict[str, torch.Tensor]],
    ):
        horizon = self.max_horizon
        if horizon <= 1:
            return {}
        midpoint = max(1, horizon // 2)
        direct = true_states[horizon]
        reanchored = self._flatten_batch_starts(
            self._gather_state(posterior, starts + midpoint)
        )
        reanchored = self._detach_state(reanchored)
        for step in range(midpoint + 1, horizon + 1):
            reanchored = dynamics.img_step(
                reanchored, action_prefix[:, step - 1], sample=False
            )
        target = self._flatten_batch_starts(
            self._gather_state(posterior, starts + horizon)
        )
        direct_p = self._project_state(dynamics, direct)
        reanchor_p = self._project_state(dynamics, reanchored)
        target_p = self._project_state(dynamics, target)

        valid = torch.ones(
            batch, num_starts, dtype=torch.bool, device=action_prefix.device
        )
        for step in range(1, horizon + 1):
            valid = valid & (~first_prefix[:, :, step - 1])
        mask = valid.reshape(-1)
        direct_target = self._masked_mean((direct_p * target_p).sum(-1), mask)
        reanchor_target = self._masked_mean((reanchor_p * target_p).sum(-1), mask)
        direct_reanchor = self._masked_mean((direct_p * reanchor_p).sum(-1), mask)
        return {
            f"dreamer_reanchored_consistency_cos_h{horizon}": direct_reanchor,
            f"dreamer_reanchor_gain_h{horizon}": reanchor_target - direct_target,
            f"dreamer_reanchored_target_cos_h{horizon}": reanchor_target,
            f"dreamer_direct_target_cos_h{horizon}": direct_target,
        }

    def _task_probe_metrics(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        rewards: torch.Tensor,
        data,
        device: torch.device,
    ):
        if not self.task_probe or "task_score" not in data or "heatmap" not in data:
            return {
                "dreamer_task_probe_available": torch.zeros((), device=device),
            }

        task_score = self._as_tensor(data["task_score"], device)
        if task_score.ndim == 3 and task_score.shape[-1] == 1:
            task_score = task_score[..., 0]
        heatmap = self._as_tensor(data["heatmap"], device)
        if heatmap.ndim == 5 and heatmap.shape[-1] == 1:
            heatmap = heatmap[..., 0]
        if heatmap.max() > 1.5:
            heatmap = heatmap / 255.0
        # Mean of top 5% pixels gives an affordance-strength statistic.
        flat_hm = heatmap.reshape(*heatmap.shape[:2], -1)
        k = max(1, int(math.ceil(0.05 * flat_hm.shape[-1])))
        aff = torch.topk(flat_hm, k=k, dim=-1).values.mean(-1)

        def robust01(x):
            flat = x.reshape(-1)
            qlo = torch.quantile(flat, 0.05)
            qhi = torch.quantile(flat, 0.95)
            return ((x - qlo) / (qhi - qlo).clamp_min(self.eps)).clamp(0.0, 1.0)

        sem_n = robust01(task_score)
        aff_n = robust01(aff)
        evidence = 0.65 * sem_n + 0.30 * aff_n
        success = rewards > 1.0e-6

        non_success = evidence[~success]
        if non_success.numel() < 4:
            return {
                "dreamer_task_probe_available": torch.ones((), device=device),
                "dreamer_task_probe_acc": torch.zeros((), device=device),
                "dreamer_task_probe_margin": torch.zeros((), device=device),
                "dreamer_task_probe_active_classes": torch.zeros((), device=device),
                "dreamer_task_evidence_span": evidence.max() - evidence.min(),
            }
        low = torch.quantile(non_success, 0.30)
        high = torch.quantile(non_success, 0.70)
        labels = torch.where(
            evidence < low,
            torch.zeros_like(evidence, dtype=torch.long),
            torch.where(
                evidence < high,
                torch.ones_like(evidence, dtype=torch.long),
                torch.full_like(evidence, 2, dtype=torch.long),
            ),
        )
        labels = torch.where(success, torch.full_like(labels, 3), labels)

        feat = self._project_state(dynamics, posterior).reshape(-1, self.projector.weight.shape[0])
        labels = labels.reshape(-1)

        centers = []
        query_feats = []
        query_labels = []
        active = []
        for cls in range(4):
            idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            # Deterministic within-class alternating split.
            support_idx = idx[::2]
            query_idx = idx[1::2]
            if query_idx.numel() == 0:
                continue
            center = F.normalize(feat[support_idx].mean(0), dim=0, eps=self.eps)
            centers.append(center)
            active.append(cls)
            query_feats.append(feat[query_idx])
            query_labels.append(torch.full(
                (query_idx.numel(),), len(active) - 1, device=device, dtype=torch.long
            ))

        if len(centers) < 2:
            return {
                "dreamer_task_probe_available": torch.ones((), device=device),
                "dreamer_task_probe_acc": torch.zeros((), device=device),
                "dreamer_task_probe_margin": torch.zeros((), device=device),
                "dreamer_task_probe_active_classes": torch.tensor(float(len(centers)), device=device),
                "dreamer_task_evidence_span": evidence.max() - evidence.min(),
            }

        centers = torch.stack(centers, 0)
        qfeat = torch.cat(query_feats, 0)
        qlabels = torch.cat(query_labels, 0)
        sims = qfeat @ centers.T
        pred = sims.argmax(-1)
        acc = (pred == qlabels).float().mean()
        correct = sims.gather(1, qlabels[:, None]).squeeze(1)
        masked = sims.clone()
        masked.scatter_(1, qlabels[:, None], -1e9)
        wrong = masked.max(-1).values
        margin = (correct - wrong).mean()
        return {
            "dreamer_task_probe_available": torch.ones((), device=device),
            "dreamer_task_probe_acc": acc,
            "dreamer_task_probe_margin": margin,
            "dreamer_task_probe_active_classes": torch.tensor(float(len(centers)), device=device),
            "dreamer_task_probe_query_fraction": torch.tensor(
                float(qfeat.shape[0]) / float(max(labels.numel(), 1)), device=device
            ),
            "dreamer_task_evidence_mean": evidence.mean(),
            "dreamer_task_evidence_std": evidence.std(unbiased=False),
            "dreamer_task_evidence_span": evidence.max() - evidence.min(),
            "dreamer_task_success_fraction": success.float().mean(),
        }

    @torch.no_grad()
    def forward(
        self,
        dynamics: nn.Module,
        reward_head: nn.Module,
        posterior: TensorDict,
        data,
    ) -> Dict[str, np.ndarray]:
        device = next(dynamics.parameters()).device
        actions = self._as_tensor(data["action"], device)
        is_first = self._as_tensor(data["is_first"], device).bool()
        rewards = self._as_tensor(data["reward"], device)
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards[..., 0]

        if actions.ndim != 3:
            return {}
        batch, time, _ = actions.shape
        starts = self._start_indices(time, device)
        if starts.numel() == 0:
            return {}
        num_starts = starts.numel()
        start_state = self._flatten_batch_starts(
            self._gather_state(posterior, starts)
        )

        # Replay convention: action[:, t] leads INTO observation/state t.
        action_steps = [actions[:, starts + o] for o in range(1, self.max_horizon + 1)]
        first_steps = [is_first[:, starts + o] for o in range(1, self.max_horizon + 1)]
        action_prefix = torch.stack(action_steps, dim=2).reshape(
            batch * num_starts, self.max_horizon, actions.shape[-1]
        )
        first_prefix = torch.stack(first_steps, dim=2).bool()

        metrics, true_states = self._counterfactual_metrics(
            dynamics, posterior, start_state, action_prefix, starts,
            first_prefix, batch, num_starts
        )
        metrics.update(self._return_metrics(
            dynamics, reward_head, posterior, start_state, action_prefix, starts,
            first_prefix, rewards, batch, num_starts
        ))
        metrics.update(self._reanchor_metrics(
            dynamics, posterior, start_state, action_prefix, starts, first_prefix,
            batch, num_starts, true_states
        ))
        metrics.update(self._task_probe_metrics(
            dynamics, posterior, rewards, data, device
        ))

        metrics["dreamer_diagnostics_active"] = torch.ones((), device=device)
        return {
            key: value.detach().float().cpu().numpy()
            if torch.is_tensor(value) else np.asarray(value, dtype=np.float32)
            for key, value in metrics.items()
        }
