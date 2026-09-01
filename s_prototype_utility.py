from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================================
# Legacy temporal-success labelling (kept only for ablation/backward comparison)
# ============================================================================

@torch.no_grad()
def steps_to_next_success(
    rewards: torch.Tensor,
    is_first: Optional[torch.Tensor] = None,
    positive_threshold: float = 1e-6,
) -> torch.Tensor:
    """Distance from each replay state to the next observed real success.

    This is retained as an optional ablation. The default v2 prototype labels do
    NOT use time-to-success, because a fixed temporal partition is not portable
    across MineDojo tasks with very different task horizons.
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
    """Legacy 4-stage time-to-success labels.

    0=Progress, 1=Near, 2=Ready, 3=Success, -1=Unknown.
    The generic v2 config defaults to `label_mode: task_evidence` instead.
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


# ============================================================================
# Generic task-evidence labelling (default v2)
# ============================================================================


def _to_bt(x: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected {name} [B,T] or [B,T,1], got {x.shape}")
    return x.float()


@torch.no_grad()
def heatmap_topk_evidence(
    heatmaps: Optional[torch.Tensor],
    topk_fraction: float = 0.05,
) -> Optional[torch.Tensor]:
    """Task-agnostic affordance evidence from a task-conditioned heatmap.

    Uses the mean of the strongest top-k pixels, independent of where the
    target appears in the image. This avoids sheep-specific assumptions such as
    "the target must be centered" and transfers to static/dynamic targets.

    Accepted shapes include [B,T,H,W], [B,T,H,W,1], and [B,T,1,H,W].
    Returns [B,T] in the original heatmap scale (normally [0,1]).
    """
    if heatmaps is None:
        return None
    x = heatmaps.float()
    if x.ndim == 5 and x.shape[-1] == 1:
        x = x[..., 0]
    elif x.ndim == 5 and x.shape[2] == 1:
        x = x[:, :, 0]
    if x.ndim != 4:
        raise ValueError(f"Expected heatmaps [B,T,H,W,(1)], got {heatmaps.shape}")

    frac = float(topk_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"topk_fraction must be in (0,1], got {frac}")

    flat = x.flatten(start_dim=2)
    k = max(1, int(round(flat.shape[-1] * frac)))
    values = torch.topk(flat, k=k, dim=-1, largest=True, sorted=False).values
    return values.mean(dim=-1)


@torch.no_grad()
def _robust_unit(
    x: Optional[torch.Tensor],
    low_quantile: float,
    high_quantile: float,
    min_signal_span: float,
) -> Tuple[Optional[torch.Tensor], bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robustly map one generic teacher signal to [0,1] within the batch.

    Quantile normalization removes task-specific numeric scale assumptions.
    A flat signal is marked inactive instead of fabricating rank differences.
    """
    if x is None:
        z = torch.tensor(0.0)
        return None, False, z, z, z

    finite = torch.isfinite(x)
    valid = x[finite]
    device = x.device
    if valid.numel() < 2:
        z = torch.zeros((), device=device)
        return torch.zeros_like(x), False, z, z, z

    q_lo = torch.quantile(valid, float(low_quantile))
    q_hi = torch.quantile(valid, float(high_quantile))
    span = q_hi - q_lo
    active = bool((span > float(min_signal_span)).item())
    if not active:
        return torch.zeros_like(x), False, q_lo, q_hi, span

    norm = ((x - q_lo) / span.clamp_min(float(min_signal_span))).clamp(0.0, 1.0)
    norm = torch.where(finite, norm, torch.zeros_like(norm))
    return norm, True, q_lo, q_hi, span


@torch.no_grad()
def build_task_evidence_labels(
    rewards: torch.Tensor,
    task_scores: Optional[torch.Tensor] = None,
    heatmaps: Optional[torch.Tensor] = None,
    intrinsic: Optional[torch.Tensor] = None,
    positive_threshold: float = 1e-6,
    semantic_weight: float = 0.65,
    affordance_weight: float = 0.30,
    intrinsic_weight: float = 0.05,
    heatmap_topk_fraction: float = 0.05,
    robust_low_quantile: float = 0.05,
    robust_high_quantile: float = 0.95,
    stage_low_quantile: float = 0.30,
    stage_high_quantile: float = 0.70,
    boundary_margin: float = 0.05,
    min_signal_span: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build generic task-evidence stages that work across MineDojo tasks.

    The labels are deliberately task-agnostic:
        0 = Low task evidence
        1 = Medium task evidence
        2 = High task evidence
        3 = Real environment success (hard anchor)
       -1 = Ambiguous / unavailable evidence

    Evidence is formed from generic task-conditioned signals already present in
    the pipeline:
        * MineCLIP task score (semantic relevance), if available;
        * task-conditioned affordance heatmap top-k response;
        * MineCLIP intrinsic progress event, with a small weight.

    Every non-flat component is robustly quantile-normalized inside the replay
    batch before combination. Therefore no sheep/tree/ore-specific numeric
    threshold is hard-coded. Real reward always overrides the pseudo stage.

    The boundary dead zones deliberately remain Unknown, reducing noisy labels
    around quantile thresholds.
    """
    rewards_bt = _to_bt(rewards, "rewards")
    assert rewards_bt is not None
    task_bt = _to_bt(task_scores, "task_scores")
    intrinsic_bt = _to_bt(intrinsic, "intrinsic")
    aff_raw = heatmap_topk_evidence(heatmaps, topk_fraction=heatmap_topk_fraction)

    if not (0.0 <= robust_low_quantile < robust_high_quantile <= 1.0):
        raise ValueError("Invalid robust quantiles")
    if not (0.0 < stage_low_quantile < stage_high_quantile < 1.0):
        raise ValueError("Invalid stage quantiles")
    if boundary_margin < 0.0:
        raise ValueError("boundary_margin must be >= 0")

    sem_norm, sem_active, sem_lo, sem_hi, sem_span = _robust_unit(
        task_bt,
        robust_low_quantile,
        robust_high_quantile,
        min_signal_span,
    )
    aff_norm, aff_active, aff_lo, aff_hi, aff_span = _robust_unit(
        aff_raw,
        robust_low_quantile,
        robust_high_quantile,
        min_signal_span,
    )

    # Intrinsic is a monotonic semantic-progress event in the existing wrapper.
    # It is intentionally weak: it nudges evidence but does not define utility.
    int_norm, int_active, int_lo, int_hi, int_span = _robust_unit(
        intrinsic_bt,
        robust_low_quantile,
        robust_high_quantile,
        min_signal_span,
    )

    components = []
    weights = []
    if sem_active and semantic_weight > 0.0:
        components.append(sem_norm)
        weights.append(float(semantic_weight))
    if aff_active and affordance_weight > 0.0:
        components.append(aff_norm)
        weights.append(float(affordance_weight))
    if int_active and intrinsic_weight > 0.0:
        components.append(int_norm)
        weights.append(float(intrinsic_weight))

    labels = torch.full_like(rewards_bt, -1, dtype=torch.long)
    evidence = torch.zeros_like(rewards_bt)

    success = rewards_bt > float(positive_threshold)

    if components:
        total_weight = sum(weights)
        evidence = sum(w * c for w, c in zip(weights, components)) / max(total_weight, 1e-8)

        finite = torch.isfinite(evidence)
        valid_evidence = evidence[finite]
        if valid_evidence.numel() >= 3:
            low_th = torch.quantile(valid_evidence, float(stage_low_quantile))
            high_th = torch.quantile(valid_evidence, float(stage_high_quantile))
            stage_span = high_th - low_th

            if bool((stage_span > float(min_signal_span)).item()):
                margin = float(boundary_margin) * stage_span

                low_mask = finite & (evidence <= (low_th - margin))
                mid_mask = finite & (evidence >= (low_th + margin)) & (
                    evidence <= (high_th - margin)
                )
                high_mask = finite & (evidence >= (high_th + margin))

                labels[low_mask] = 0
                labels[mid_mask] = 1
                labels[high_mask] = 2
            else:
                low_th = torch.zeros((), device=rewards_bt.device)
                high_th = torch.zeros((), device=rewards_bt.device)
                stage_span = torch.zeros((), device=rewards_bt.device)
        else:
            low_th = torch.zeros((), device=rewards_bt.device)
            high_th = torch.zeros((), device=rewards_bt.device)
            stage_span = torch.zeros((), device=rewards_bt.device)
    else:
        low_th = torch.zeros((), device=rewards_bt.device)
        high_th = torch.zeros((), device=rewards_bt.device)
        stage_span = torch.zeros((), device=rewards_bt.device)

    # Real success is the highest-authority anchor and can never be overwritten
    # by a pseudo task-evidence class.
    labels[success] = 3

    metrics: Dict[str, torch.Tensor] = {
        "s_proto_teacher_semantic_active": torch.tensor(
            float(sem_active), device=rewards_bt.device
        ),
        "s_proto_teacher_affordance_active": torch.tensor(
            float(aff_active), device=rewards_bt.device
        ),
        "s_proto_teacher_intrinsic_active": torch.tensor(
            float(int_active), device=rewards_bt.device
        ),
        "s_proto_teacher_semantic_qlo": sem_lo.to(rewards_bt.device),
        "s_proto_teacher_semantic_qhi": sem_hi.to(rewards_bt.device),
        "s_proto_teacher_semantic_span": sem_span.to(rewards_bt.device),
        "s_proto_teacher_affordance_qlo": aff_lo.to(rewards_bt.device),
        "s_proto_teacher_affordance_qhi": aff_hi.to(rewards_bt.device),
        "s_proto_teacher_affordance_span": aff_span.to(rewards_bt.device),
        "s_proto_teacher_intrinsic_qlo": int_lo.to(rewards_bt.device),
        "s_proto_teacher_intrinsic_qhi": int_hi.to(rewards_bt.device),
        "s_proto_teacher_intrinsic_span": int_span.to(rewards_bt.device),
        "s_proto_evidence_mean": evidence.mean().detach(),
        "s_proto_evidence_std": evidence.std(unbiased=False).detach(),
        "s_proto_stage_low_threshold": low_th.detach(),
        "s_proto_stage_high_threshold": high_th.detach(),
        "s_proto_stage_threshold_span": stage_span.detach(),
        "s_proto_label_low_fraction": (labels == 0).float().mean().detach(),
        "s_proto_label_mid_fraction": (labels == 1).float().mean().detach(),
        "s_proto_label_high_fraction": (labels == 2).float().mean().detach(),
        "s_proto_label_success_fraction": (labels == 3).float().mean().detach(),
        "s_proto_label_unknown_fraction": (labels < 0).float().mean().detach(),
    }
    return labels, evidence, metrics


# ============================================================================
# Prototype bank
# ============================================================================


class PrototypeUtilityBank(nn.Module):
    """EMA Prototypical-Network bank over the existing fixed S5A projection.

    Default generic class semantics:
        0 = Low task evidence
        1 = Medium task evidence
        2 = High task evidence
        3 = Real success

    Real posterior-S features are detached support examples. S5A-predicted
    future-S features are differentiable queries. Therefore the prototype loss
    shapes the existing S dynamics/representation without introducing another
    trainable utility encoder.

    `utility()` remains diagnostic only in v2 and is NOT added to actor reward.
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
            torch.tensor([0.00, 0.33, 0.66, 1.00], dtype=torch.float32),
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

            self.prototypes[cls].copy_(merged.to(dtype=self.prototypes.dtype))
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
        values = self.class_values.to(device=probs.device, dtype=probs.dtype)
        return (probs * values).sum(dim=-1)

    def metrics(self) -> Dict[str, torch.Tensor]:
        return {
            "s_proto_initialized": self.initialized.float().sum().detach(),
            "s_proto_support_low": self.support_count[0].float().detach(),
            "s_proto_support_mid": self.support_count[1].float().detach(),
            "s_proto_support_high": self.support_count[2].float().detach(),
            "s_proto_support_success": self.support_count[3].float().detach(),
        }
