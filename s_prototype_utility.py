from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def steps_to_next_success(
    rewards: torch.Tensor,
    is_first: Optional[torch.Tensor] = None,
    positive_threshold: float = 1e-6,
) -> torch.Tensor:
    """Distance from each replay state to the next observed real success.

    `reward == 0` is NOT treated as negative.  If no future success is visible
    inside the current episode/chunk, the label is unknown (-1).
    """
    if rewards.ndim == 3 and rewards.shape[-1] == 1:
        rewards = rewards[..., 0]
    if rewards.ndim != 2:
        raise ValueError(f"Expected rewards [B,T] or [B,T,1], got {rewards.shape}")

    rewards = rewards.float()
    batch, time = rewards.shape
    device = rewards.device

    if is_first is not None:
        if is_first.ndim == 3 and is_first.shape[-1] == 1:
            is_first = is_first[..., 0]
        if is_first.shape != rewards.shape:
            raise ValueError(
                f"is_first {is_first.shape} does not match rewards {rewards.shape}"
            )
        is_first = is_first.bool()

    inf = time + 1000
    next_dist = torch.full((batch,), inf, dtype=torch.long, device=device)
    distance = torch.full((batch, time), -1, dtype=torch.long, device=device)

    for t in reversed(range(time)):
        # state t must not inherit a success from a new episode starting at t+1.
        if is_first is not None and t < time - 1:
            reset_after_t = is_first[:, t + 1]
            next_dist = torch.where(
                reset_after_t,
                torch.full_like(next_dist, inf),
                next_dist,
            )

        success_now = rewards[:, t].abs() > float(positive_threshold)
        next_dist = torch.where(
            success_now,
            torch.zeros_like(next_dist),
            next_dist + 1,
        )
        valid = next_dist <= time
        distance[:, t] = torch.where(
            valid,
            next_dist,
            torch.full_like(next_dist, -1),
        )

    return distance


@torch.no_grad()
def build_progress_labels(
    rewards: torch.Tensor,
    is_first: Optional[torch.Tensor] = None,
    positive_threshold: float = 1e-6,
    ready_steps: int = 4,
    near_steps: int = 15,
    progress_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build four ordinal few-shot task-stage labels.

    0 = Progress: future success is farther than `near_steps`
    1 = Near:     success within (ready_steps, near_steps]
    2 = Ready:    success within [1, ready_steps]
    3 = Success:  current replay state carries a real environment success reward
   -1 = Unknown:  no future success is observed in this episode/chunk

    If `progress_steps` is not None, only states within that many steps from
    success are assigned Progress; earlier states stay Unknown.
    """
    ready_steps = int(ready_steps)
    near_steps = int(near_steps)
    if ready_steps < 1:
        raise ValueError("ready_steps must be >= 1")
    if near_steps <= ready_steps:
        raise ValueError("near_steps must be > ready_steps")

    distance = steps_to_next_success(
        rewards,
        is_first=is_first,
        positive_threshold=positive_threshold,
    )
    labels = torch.full_like(distance, -1)

    labels[distance == 0] = 3

    ready = (distance >= 1) & (distance <= ready_steps)
    labels[ready] = 2

    near = (distance > ready_steps) & (distance <= near_steps)
    labels[near] = 1

    progress = distance > near_steps
    if progress_steps is not None:
        progress = progress & (distance <= int(progress_steps))
    labels[progress] = 0

    return labels, distance


class PrototypeUtilityBank(nn.Module):
    """EMA Prototypical-Network bank over the existing fixed S5A projection.

    Real posterior-S features are detached support examples.  S5A-predicted
    future-S features are differentiable queries.  Therefore prototype loss
    shapes the existing S dynamics/representation without introducing another
    trainable utility encoder.

    v1 is representation supervision only: `utility()` is for diagnostics and
    is NOT added to actor imagined reward.
    """

    def __init__(
        self,
        feature_dim: int,
        temperature: float = 0.10,
        prototype_ema: float = 0.95,
        eps: float = 1e-8,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError(feature_dim)
        if temperature <= 0.0:
            raise ValueError(temperature)
        if not (0.0 <= prototype_ema < 1.0):
            raise ValueError(prototype_ema)

        self.feature_dim = feature_dim
        self.temperature = float(temperature)
        self.prototype_ema = float(prototype_ema)
        self.eps = float(eps)
        self.num_classes = 4

        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.feature_dim),
            persistent=True,
        )
        self.register_buffer(
            "initialized",
            torch.zeros(self.num_classes, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "support_count",
            torch.zeros(self.num_classes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "class_values",
            torch.tensor([0.25, 0.50, 0.75, 1.00], dtype=torch.float32),
            persistent=True,
        )

    @torch.no_grad()
    def update(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if support_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Support feature dim {support_features.shape[-1]} "
                f"!= {self.feature_dim}"
            )

        feats = support_features.reshape(-1, self.feature_dim).float()
        labels = support_labels.reshape(-1)
        if feats.shape[0] != labels.shape[0]:
            raise ValueError("Support feature/label count mismatch")

        feats = F.normalize(feats, dim=-1, eps=self.eps)
        added = {}

        for cls in range(self.num_classes):
            mask = labels == cls
            count = mask.sum()
            added[f"prototype_added_c{cls}"] = count.detach()
            if not bool((count > 0).item()):
                continue

            new_proto = F.normalize(
                feats[mask].mean(dim=0),
                dim=-1,
                eps=self.eps,
            )

            if not bool(self.initialized[cls].item()):
                merged = new_proto
                self.initialized[cls] = True
            else:
                old = self.prototypes[cls].float()
                merged = (
                    self.prototype_ema * old
                    + (1.0 - self.prototype_ema) * new_proto
                )
                merged = F.normalize(merged, dim=-1, eps=self.eps)

            self.prototypes[cls].copy_(
                merged.to(dtype=self.prototypes.dtype)
            )
            self.support_count[cls] += count.to(self.support_count.dtype)

        return added

    def logits(self, query_features: torch.Tensor) -> torch.Tensor:
        if query_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Query feature dim {query_features.shape[-1]} "
                f"!= {self.feature_dim}"
            )

        query = F.normalize(query_features.float(), dim=-1, eps=self.eps)
        proto = F.normalize(self.prototypes.float(), dim=-1, eps=self.eps)
        logits = torch.matmul(query, proto.transpose(0, 1)) / self.temperature

        invalid = ~self.initialized
        return logits.masked_fill(invalid, -1.0e9)

    def loss(
        self,
        query_features: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat_query = query_features.reshape(-1, self.feature_dim)
        flat_labels = target_labels.reshape(-1).long()
        if flat_query.shape[0] != flat_labels.shape[0]:
            raise ValueError("Query feature/label count mismatch")

        zero = flat_query.sum() * 0.0
        device = flat_query.device

        # ProtoNet needs at least two available stages to define a non-trivial
        # metric-learning classification objective.
        if int(self.initialized.sum().item()) < 2:
            return zero, {
                "acc": torch.zeros((), device=device),
                "num": torch.zeros((), device=device),
            }

        valid = flat_labels >= 0
        safe = flat_labels.clamp(min=0, max=self.num_classes - 1)
        valid = valid & self.initialized[safe]

        if not bool(valid.any().item()):
            return zero, {
                "acc": torch.zeros((), device=device),
                "num": torch.zeros((), device=device),
            }

        query = flat_query[valid]
        labels = flat_labels[valid]
        logits = self.logits(query)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()

        return loss, {
            "acc": acc.detach(),
            "num": valid.float().sum().detach(),
        }

    @torch.no_grad()
    def utility(self, query_features: torch.Tensor) -> torch.Tensor:
        logits = self.logits(query_features)
        probs = torch.softmax(logits, dim=-1)
        values = self.class_values.to(
            device=probs.device,
            dtype=probs.dtype,
        )
        return (probs * values).sum(dim=-1)

    def metrics(self) -> Dict[str, torch.Tensor]:
        return {
            "s_proto_initialized": self.initialized.float().sum().detach(),
            "s_proto_support_progress": self.support_count[0].float().detach(),
            "s_proto_support_near": self.support_count[1].float().detach(),
            "s_proto_support_ready": self.support_count[2].float().detach(),
            "s_proto_support_success": self.support_count[3].float().detach(),
        }
