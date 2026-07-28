"""Open-loop multi-step RSSM latent consistency for Experiment 5A.

This module deliberately does not add a new latent branch, object token, semantic
teacher, reward, or actor/critic input. It only regularizes the existing RSSM so
that an open-loop prior rollout from a posterior start state remains consistent
with future posterior states observed in replay.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


TensorDict = Mapping[str, torch.Tensor]


class FixedRandomProjector(nn.Module):
    """Deterministic frozen projection that does not consume the global RNG.

    A fixed projection reduces the 5k+ dimensional Dreamer feature to a smaller
    comparison space without adding a trainable target network or changing the
    initialization of the original Dreamer modules.
    """

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


class MultiStepRSSMConsistency(nn.Module):
    """Sparse-horizon open-loop consistency objective for a standard RSSM.

    For each replay sequence, a small deterministic set of start positions is
    chosen. Starting from the posterior state at each position, the original
    RSSM prior is rolled forward with replay actions and compared against future
    posterior states at selected horizons.

    The comparison uses distribution means/probabilities rather than sampled
    one-hot states, while the rollout itself uses the RSSM's straight-through
    stochastic samples so gradients propagate through the actual transition.
    """

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
    def _state_feature(dynamics: nn.Module, state: TensorDict) -> torch.Tensor:
        """Smooth feature from an RSSM state distribution.

        Standard unified Dreamer RSSM:
          discrete: [softmax(logit), deter]
          continuous: [mean, deter]

        A legacy split state is supported only to produce a clear error-tolerant
        patch, but Experiment 5A should be run on the unified RSSM baseline.
        """
        if "deter" in state:
            deter = state["deter"]
            if getattr(dynamics, "_discrete", False):
                stoch = torch.softmax(state["logit"], dim=-1)
                stoch = stoch.reshape(*stoch.shape[:-2], -1)
            else:
                stoch = state["mean"]
            return torch.cat([stoch, deter], dim=-1)

        if "deter_s" in state and "deter_z" in state:
            parts = []
            if getattr(dynamics, "_discrete", False):
                for prefix in ("s", "z"):
                    key = f"{prefix}_logit"
                    if key not in state:
                        raise KeyError(
                            f"Missing {key}; Experiment 5A expects a unified RSSM."
                        )
                    probs = torch.softmax(state[key], dim=-1)
                    parts.append(probs.reshape(*probs.shape[:-2], -1))
                    parts.append(state[f"deter_{prefix}"])
            else:
                for prefix in ("s", "z"):
                    parts.append(state[f"{prefix}_mean"])
                    parts.append(state[f"deter_{prefix}"])
            return torch.cat(parts, dim=-1)

        raise KeyError(f"Unrecognized RSSM state keys: {tuple(state.keys())}")

    @staticmethod
    def _gather_state(state: TensorDict, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Gather [B,T,...] state entries at shared time indices -> [B,S,...]."""
        return {key: value[:, indices] for key, value in state.items()}

    @staticmethod
    def _flatten_batch_starts(state: TensorDict) -> Dict[str, torch.Tensor]:
        """[B,S,...] -> [B*S,...]."""
        out: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            out[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        return out

    def _start_indices(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        usable = sequence_length - self.max_horizon
        if usable <= 0:
            raise ValueError(
                f"batch_length={sequence_length} must exceed max_horizon={self.max_horizon}."
            )
        count = min(self.starts_per_sequence, usable)
        if count == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        # Deterministic, evenly-spaced positions avoid consuming the global RNG,
        # which keeps seed-matched baseline initialization comparable.
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
            return zero, {"multistep_valid_fraction": zero.detach()}

        starts = self._start_indices(time, actions.device)
        num_starts = starts.numel()

        start_state = self._gather_state(posterior, starts)
        current = self._flatten_batch_starts(start_state)

        # action[:, t] is the action that leads into observation/state t in this
        # codebase. From state t, the first future transition therefore uses
        # action[:, t + 1].
        action_steps = []
        first_steps = []
        for offset in range(1, self.max_horizon + 1):
            action_steps.append(actions[:, starts + offset])       # [B,S,A]
            first_steps.append(is_first[:, starts + offset])      # [B,S]
        action_prefix = torch.stack(action_steps, dim=2)           # [B,S,H,A]
        first_prefix = torch.stack(first_steps, dim=2).bool()      # [B,S,H]
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
            current = dynamics.img_step(
                current, action_prefix[:, step - 1], sample=True
            )
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])

            if step not in horizon_to_index:
                continue

            target_indices = starts + step
            target_state = self._gather_state(posterior, target_indices)
            target_state = self._flatten_batch_starts(target_state)

            pred_feat = self._state_feature(dynamics, current)
            target_feat = self._state_feature(dynamics, target_state)

            pred_proj = self.projector(pred_feat.float())
            with torch.no_grad():
                target_proj = self.projector(target_feat.float())

            pred_proj = F.normalize(pred_proj, dim=-1, eps=self.eps)
            target_proj = F.normalize(target_proj, dim=-1, eps=self.eps)
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

            metrics[f"multistep_loss_h{step}"] = horizon_loss.detach()
            metrics[f"multistep_cosine_h{step}"] = horizon_cosine.detach()
            metrics[f"multistep_valid_h{step}"] = valid_fraction.detach()

        total = weighted_loss / weight_total.clamp_min(self.eps)
        metrics["multistep_loss"] = total.detach()
        metrics["multistep_starts"] = torch.tensor(
            float(num_starts), device=actions.device
        )
        metrics["multistep_target_std"] = target_proj.detach().std()
        return total, metrics
