"""S-only open-loop multi-step RSSM consistency for ISO3.

Adapted from the user's successful DreamerV3 Experiment 5A. The objective is
restricted to the controllable S branch: starting from replay posterior S_t,
the existing S prior transition is rolled forward with replay action prefixes
and compared to future posterior S states. Z is neither an input to the loss nor
updated by the auxiliary rollout.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

TensorDict = Mapping[str, torch.Tensor]


class FixedRandomProjector(nn.Module):
    """Frozen random projection that does not consume the global RNG."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 314159):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError((input_dim, output_dim))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weight = torch.randn(output_dim, input_dim, generator=generator)
        weight = weight / math.sqrt(float(output_dim))
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight)


class SOnlyMultiStepRSSMConsistency(nn.Module):
    """5A-style open-loop consistency applied only to ISO3's S branch."""

    def __init__(
        self,
        feat_dim: int,
        horizons: Sequence[int] = (1, 2, 4, 8, 15),
        horizon_weights: Sequence[float] = (1.0, 1.0, 0.75, 0.5, 0.25),
        projection_dim: int = 512,
        starts_per_sequence: int = 4,
        projection_seed: int = 314159,
        eps: float = 1e-8,
    ):
        super().__init__()
        horizons = tuple(int(x) for x in horizons)
        weights = tuple(float(x) for x in horizon_weights)
        if not horizons or any(x <= 0 for x in horizons):
            raise ValueError(f"Invalid horizons: {horizons}")
        if sorted(set(horizons)) != list(horizons):
            raise ValueError(f"Horizons must be unique and sorted: {horizons}")
        if len(horizons) != len(weights):
            raise ValueError((horizons, weights))
        if any(x < 0.0 for x in weights) or sum(weights) <= 0.0:
            raise ValueError(f"Invalid horizon weights: {weights}")
        if starts_per_sequence <= 0:
            raise ValueError(starts_per_sequence)

        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.starts_per_sequence = int(starts_per_sequence)
        self.eps = float(eps)
        self.projector = FixedRandomProjector(
            int(feat_dim), int(projection_dim), int(projection_seed)
        )
        self.register_buffer(
            "horizon_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _s_feature(dynamics: nn.Module, state: TensorDict) -> torch.Tensor:
        """Smooth S feature: [posterior/prior probabilities, deter_s]."""
        deter = state["deter_s"]
        if getattr(dynamics, "_discrete", False):
            logits = state["s_logit"]
            stoch = torch.softmax(logits, dim=-1)
            stoch = stoch.reshape(*stoch.shape[:-2], -1)
        else:
            stoch = state["s_mean"]
        return torch.cat([stoch, deter], dim=-1)

    @staticmethod
    def _gather_s_state(
        state: TensorDict, indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        keys = ["stoch_s", "deter_s"]
        if "s_logit" in state:
            keys.append("s_logit")
        else:
            keys.extend(["s_mean", "s_std"])
        return {key: state[key][:, indices] for key in keys}

    @staticmethod
    def _flatten_batch_starts(
        state: TensorDict,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            out[key] = value.reshape(
                value.shape[0] * value.shape[1], *value.shape[2:]
            )
        return out

    def _start_indices(
        self, sequence_length: int, device: torch.device
    ) -> torch.Tensor:
        usable = sequence_length - self.max_horizon
        if usable <= 0:
            raise ValueError(
                f"batch_length={sequence_length} must exceed "
                f"max_horizon={self.max_horizon}."
            )
        count = min(self.starts_per_sequence, usable)
        if count == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(
            0, usable - 1, steps=count, device=device
        ).round().long()

    def forward(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        actions: torch.Tensor,
        is_first: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,T,A], got {actions.shape}")
        batch, time, _ = actions.shape
        if time <= self.max_horizon:
            zero = actions.sum() * 0.0
            return zero, {"s_multistep_valid_fraction": zero.detach()}

        starts = self._start_indices(time, actions.device)
        num_starts = starts.numel()

        start_state = self._gather_s_state(posterior, starts)
        current = self._flatten_batch_starts(start_state)

        # In this replay convention action[:, t] leads INTO state/observation t.
        # Therefore the first transition from state t uses action[:, t + 1].
        action_steps = []
        first_steps = []
        for offset in range(1, self.max_horizon + 1):
            action_steps.append(actions[:, starts + offset])
            first_steps.append(is_first[:, starts + offset])
        action_prefix = torch.stack(action_steps, dim=2)
        first_prefix = torch.stack(first_steps, dim=2).bool()
        action_prefix = action_prefix.reshape(
            batch * num_starts, self.max_horizon, actions.shape[-1]
        )

        horizon_to_index = {h: i for i, h in enumerate(self.horizons)}
        weighted_loss = actions.sum() * 0.0
        weight_total = actions.sum() * 0.0
        metrics: Dict[str, torch.Tensor] = {}
        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=actions.device
        )

        for step in range(1, self.max_horizon + 1):
            current = dynamics.img_step_s(
                current, action_prefix[:, step - 1], sample=True
            )
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])

            if step not in horizon_to_index:
                continue

            target_indices = starts + step
            target_state = self._gather_s_state(posterior, target_indices)
            target_state = self._flatten_batch_starts(target_state)

            pred_feat = self._s_feature(dynamics, current)
            target_feat = self._s_feature(dynamics, target_state)

            pred_raw = self.projector(pred_feat.float())
            with torch.no_grad():
                target_raw = self.projector(target_feat.float())

            pred_proj = F.normalize(pred_raw, dim=-1, eps=self.eps)
            target_proj = F.normalize(target_raw, dim=-1, eps=self.eps)
            cosine = torch.sum(pred_proj * target_proj, dim=-1)
            loss_vector = 1.0 - cosine

            mask = valid_prefix.reshape(-1).to(loss_vector.dtype)
            denom = mask.sum().clamp_min(1.0)
            horizon_loss = (loss_vector * mask).sum() / denom
            horizon_cosine = (cosine * mask).sum() / denom
            valid_fraction = mask.mean()

            weight = self.horizon_weights[horizon_to_index[step]].to(
                device=horizon_loss.device, dtype=horizon_loss.dtype
            )
            weighted_loss = weighted_loss + weight * horizon_loss
            weight_total = weight_total + weight

            metrics[f"s_multistep_loss_h{step}"] = horizon_loss.detach()
            metrics[f"s_multistep_cosine_h{step}"] = horizon_cosine.detach()
            metrics[f"s_multistep_valid_h{step}"] = valid_fraction.detach()
            # Unlike the old 5A target_std (computed after L2 normalization),
            # this diagnostic is measured before normalization and is meaningful.
            metrics[f"s_multistep_target_raw_std_h{step}"] = (
                target_raw.detach().std()
            )

        total = weighted_loss / weight_total.clamp_min(self.eps)
        metrics["s_multistep_loss"] = total.detach()
        metrics["s_multistep_starts"] = torch.tensor(
            float(num_starts), device=actions.device
        )
        return total, metrics
