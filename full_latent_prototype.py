"""Generic Task-Evidence Prototype Utility on the *single-stream* Dreamer latent.

This module is the controlled ablation counterpart of the existing S-branch
prototype utility.  It deliberately keeps the same teacher construction,
prototype bank, fixed random projection, horizons, and loss weighting, but
replaces the controllable S feature with the standard Dreamer RSSM full latent
feature.

Important design constraints:
  * no Dual S/Z split;
  * no S-Aff reconstruction loss;
  * no S5A cosine/consistency loss;
  * no Balanced Outcome loss;
  * no extra trainable encoder/projector;
  * prototype utility is auxiliary world-model supervision only and is NOT
    added to the actor reward.

The only differentiable auxiliary path is:
    posterior full latent x_t --Dreamer RSSM prior + replay actions-->
    x_hat_{t+k} --fixed random projector--> query
    --prototype CE--> Generic Task-Evidence class.

Support prototypes are updated from detached replay posterior full latents.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from s_prototype_utility import (
    PrototypeUtilityBank,
    build_progress_labels,
    build_task_evidence_labels,
)

TensorDict = Mapping[str, torch.Tensor]


class FixedRandomProjector(nn.Module):
    """Frozen Gaussian projection that does not perturb the global torch RNG."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 314159):
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
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


class FullLatentPrototypeUtility(nn.Module):
    """Generic ProtoNet auxiliary loss on standard Dreamer RSSM latents.

    The class semantics match the GenericTask S-branch version exactly:
      0 = Low task evidence
      1 = Mid task evidence
      2 = High task evidence
      3 = real environment Success
     -1 = Unknown / ambiguous pseudo-label

    The posterior support path is no-grad.  The predicted future query path is
    differentiable and therefore trains the ordinary Dreamer RSSM through its
    native ``img_step`` transition.  There is intentionally no S5A consistency
    objective in this class.
    """

    def __init__(
        self,
        feat_dim: int,
        horizons: Sequence[int] = (1, 2, 4, 8, 15),
        horizon_weights: Sequence[float] = (1.0, 1.0, 0.75, 0.5, 0.25),
        projection_dim: int = 512,
        starts_per_sequence: int = 4,
        projection_seed: int = 314159,
        temperature: float = 0.10,
        prototype_ema: float = 0.95,
        label_mode: str = "task_evidence",
        positive_threshold: float = 1.0e-6,
        semantic_weight: float = 0.65,
        affordance_weight: float = 0.30,
        intrinsic_weight: float = 0.05,
        heatmap_topk_fraction: float = 0.05,
        robust_low_quantile: float = 0.05,
        robust_high_quantile: float = 0.95,
        stage_low_quantile: float = 0.30,
        stage_high_quantile: float = 0.70,
        boundary_margin: float = 0.05,
        min_signal_span: float = 1.0e-4,
        # Legacy temporal-success mode, retained only for controlled ablation.
        ready_steps: int = 4,
        near_steps: int = 15,
        progress_steps: Optional[int] = None,
        diagnostics_enabled: bool = True,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        horizons = tuple(int(h) for h in horizons)
        weights = tuple(float(w) for w in horizon_weights)
        if not horizons or any(h <= 0 for h in horizons):
            raise ValueError(f"Invalid horizons: {horizons}")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError(f"Horizons must be sorted and unique: {horizons}")
        if len(horizons) != len(weights):
            raise ValueError((horizons, weights))
        if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
            raise ValueError(f"Invalid horizon_weights: {weights}")
        if int(starts_per_sequence) <= 0:
            raise ValueError(starts_per_sequence)

        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.starts_per_sequence = int(starts_per_sequence)
        self.eps = float(eps)
        self.diagnostics_enabled = bool(diagnostics_enabled)

        self.projector = FixedRandomProjector(
            int(feat_dim), int(projection_dim), int(projection_seed)
        )
        self.prototype_bank = PrototypeUtilityBank(
            feature_dim=int(projection_dim),
            temperature=float(temperature),
            prototype_ema=float(prototype_ema),
            eps=self.eps,
        )
        self.register_buffer(
            "horizon_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=True,
        )

        self.label_mode = str(label_mode).lower()
        if self.label_mode not in {"task_evidence", "temporal_success"}:
            raise ValueError(
                "label_mode must be task_evidence or temporal_success, got "
                f"{self.label_mode}"
            )
        self.positive_threshold = float(positive_threshold)
        self.semantic_weight = float(semantic_weight)
        self.affordance_weight = float(affordance_weight)
        self.intrinsic_weight = float(intrinsic_weight)
        self.heatmap_topk_fraction = float(heatmap_topk_fraction)
        self.robust_low_quantile = float(robust_low_quantile)
        self.robust_high_quantile = float(robust_high_quantile)
        self.stage_low_quantile = float(stage_low_quantile)
        self.stage_high_quantile = float(stage_high_quantile)
        self.boundary_margin = float(boundary_margin)
        self.min_signal_span = float(min_signal_span)
        self.ready_steps = int(ready_steps)
        self.near_steps = int(near_steps)
        self.progress_steps = None if progress_steps is None else int(progress_steps)

    @staticmethod
    def _feature(dynamics: nn.Module, state: TensorDict) -> torch.Tensor:
        """Smooth full Dreamer feature: [distribution statistic, deter].

        For categorical RSSM states we use softmax probabilities rather than a
        sampled one-hot vector for the metric/prototype feature.  This mirrors
        the existing S-only prototype implementation and reduces label noise.
        The recurrent transition itself still consumes the normal RSSM sampled
        stochastic state, exactly as standard Dreamer training does.
        """
        deter = state["deter"]
        if getattr(dynamics, "_discrete", False):
            logits = state["logit"]
            stoch = torch.softmax(logits, dim=-1)
            stoch = stoch.reshape(*stoch.shape[:-2], -1)
        else:
            stoch = state["mean"]
        return torch.cat([stoch, deter], dim=-1)

    @staticmethod
    def _gather_state(state: TensorDict, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        keys = ["stoch", "deter"]
        if "logit" in state:
            keys.append("logit")
        else:
            keys.extend(["mean", "std"])
        return {key: state[key][:, indices] for key in keys}

    @staticmethod
    def _flatten_batch_starts(state: TensorDict) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            out[key] = value.reshape(
                value.shape[0] * value.shape[1], *value.shape[2:]
            )
        return out

    @staticmethod
    def _detach_state(state: TensorDict) -> Dict[str, torch.Tensor]:
        return {key: value.detach() for key, value in state.items()}

    def _start_indices(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        usable = int(sequence_length) - self.max_horizon
        if usable <= 0:
            raise ValueError(
                f"batch_length={sequence_length} must exceed max_horizon={self.max_horizon}"
            )
        count = min(self.starts_per_sequence, usable)
        if count == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(0, usable - 1, steps=count, device=device).round().long()

    def _project_feature(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.projector(feature.float())
        return F.normalize(raw, dim=-1, eps=self.eps)

    @staticmethod
    def _rename_teacher_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expose backend-neutral names while retaining exact teacher semantics."""
        out: Dict[str, torch.Tensor] = {}
        for key, value in metrics.items():
            if key.startswith("s_proto_"):
                key = "proto_" + key[len("s_proto_"):]
            out[key] = value
        return out

    @torch.no_grad()
    def _prepare_support(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        rewards: torch.Tensor,
        is_first: torch.Tensor,
        task_scores: Optional[torch.Tensor],
        heatmaps: Optional[torch.Tensor],
        intrinsic: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.label_mode == "task_evidence":
            labels, _, teacher_metrics = build_task_evidence_labels(
                rewards=rewards,
                task_scores=task_scores,
                heatmaps=heatmaps,
                intrinsic=intrinsic,
                positive_threshold=self.positive_threshold,
                semantic_weight=self.semantic_weight,
                affordance_weight=self.affordance_weight,
                intrinsic_weight=self.intrinsic_weight,
                heatmap_topk_fraction=self.heatmap_topk_fraction,
                robust_low_quantile=self.robust_low_quantile,
                robust_high_quantile=self.robust_high_quantile,
                stage_low_quantile=self.stage_low_quantile,
                stage_high_quantile=self.stage_high_quantile,
                boundary_margin=self.boundary_margin,
                min_signal_span=self.min_signal_span,
            )
            metrics = self._rename_teacher_metrics(teacher_metrics)
            class_names = ("low", "mid", "high", "success")
        else:
            labels, _ = build_progress_labels(
                rewards=rewards,
                is_first=is_first,
                positive_threshold=self.positive_threshold,
                ready_steps=self.ready_steps,
                near_steps=self.near_steps,
                progress_steps=self.progress_steps,
            )
            metrics = {}
            class_names = ("progress", "near", "ready", "success")

        posterior_feat = self._feature(dynamics, posterior)
        support_proj = self._project_feature(posterior_feat)
        added = self.prototype_bank.update(
            support_features=support_proj.detach(),
            support_labels=labels.detach(),
        )

        bank_metrics = self.prototype_bank.metrics()
        for key, value in bank_metrics.items():
            if key.startswith("s_proto_"):
                key = "proto_" + key[len("s_proto_"):]
            metrics[key] = value.detach()
        for key, value in added.items():
            metrics["proto_" + key] = value.detach()

        known = labels >= 0
        metrics["proto_known_fraction"] = known.float().mean().detach()
        metrics["proto_unknown_fraction"] = (~known).float().mean().detach()
        metrics["proto_label_mode_task_evidence"] = torch.tensor(
            float(self.label_mode == "task_evidence"), device=rewards.device
        )
        for cls, name in enumerate(class_names):
            metrics[f"proto_batch_{name}_count"] = (labels == cls).float().sum().detach()
        return labels, metrics

    @torch.no_grad()
    def _deterministic_action_diagnostics(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        starts: torch.Tensor,
        first_prefix: torch.Tensor,
        batch: int,
        num_starts: int,
    ) -> Dict[str, torch.Tensor]:
        """Counterfactual diagnostics on the ordinary Dreamer full latent.

        They are logging only and use ``sample=False`` so diagnostics do not
        consume RNG or alter the training trajectory.
        """
        if not self.diagnostics_enabled:
            return {}

        capture = set(self.horizons)

        def rollout(actions: torch.Tensor):
            current = self._detach_state(start_state)
            states: Dict[int, Dict[str, torch.Tensor]] = {}
            for step in range(1, self.max_horizon + 1):
                current = dynamics.img_step(current, actions[:, step - 1], sample=False)
                if step in capture:
                    states[step] = self._detach_state(current)
            return states

        true_states = rollout(action_prefix.detach())
        if action_prefix.shape[0] > 1:
            shuffled = torch.roll(action_prefix.detach(), shifts=1, dims=0)
        else:
            shuffled = torch.flip(action_prefix.detach(), dims=[1])
        shuffled_states = rollout(shuffled)

        metrics: Dict[str, torch.Tensor] = {}
        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=action_prefix.device
        )
        for step in range(1, self.max_horizon + 1):
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])
            if step not in capture:
                continue
            target_indices = starts + step
            target_state = self._flatten_batch_starts(
                self._gather_state(posterior, target_indices)
            )
            target_proj = self._project_feature(self._feature(dynamics, target_state))
            true_proj = self._project_feature(self._feature(dynamics, true_states[step]))
            shuf_proj = self._project_feature(self._feature(dynamics, shuffled_states[step]))
            mask = valid_prefix.reshape(-1).to(true_proj.dtype)
            denom = mask.sum().clamp_min(1.0)
            true_cos = ((true_proj * target_proj).sum(-1) * mask).sum() / denom
            shuf_cos = ((shuf_proj * target_proj).sum(-1) * mask).sum() / denom
            response = (1.0 - (true_proj * shuf_proj).sum(-1))
            response = (response * mask).sum() / denom
            metrics[f"proto_cf_true_target_cos_h{step}"] = true_cos.detach()
            metrics[f"proto_cf_shuffled_target_cos_h{step}"] = shuf_cos.detach()
            metrics[f"proto_cf_margin_h{step}"] = (true_cos - shuf_cos).detach()
            metrics[f"proto_response_gap_h{step}"] = response.detach()
            metrics[f"proto_diag_valid_h{step}"] = mask.mean().detach()
        return metrics

    def forward(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        rewards: torch.Tensor,
        task_scores: Optional[torch.Tensor] = None,
        heatmaps: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,T,A], got {actions.shape}")
        batch, time, _ = actions.shape
        zero = actions.sum() * 0.0
        if time <= self.max_horizon:
            return zero, {
                "proto_valid_fraction": zero.detach(),
                "proto_loss": zero.detach(),
            }

        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards[..., 0]
        if rewards.ndim != 2 or rewards.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"Expected rewards [B,T] matching actions, got {rewards.shape}"
            )

        labels, metrics = self._prepare_support(
            dynamics=dynamics,
            posterior=posterior,
            rewards=rewards,
            is_first=is_first,
            task_scores=task_scores,
            heatmaps=heatmaps,
            intrinsic=intrinsic,
        )

        starts = self._start_indices(time, actions.device)
        num_starts = int(starts.numel())
        start_state = self._gather_state(posterior, starts)
        current = self._flatten_batch_starts(start_state)
        start_state_flat = current

        # Replay convention: action[:, t] leads into observation/state t.
        # Thus the first transition from state t uses action[:, t+1].
        action_steps = []
        first_steps = []
        for offset in range(1, self.max_horizon + 1):
            action_steps.append(actions[:, starts + offset])
            first_steps.append(is_first[:, starts + offset])
        action_prefix = torch.stack(action_steps, dim=2).reshape(
            batch * num_starts, self.max_horizon, actions.shape[-1]
        )
        first_prefix = torch.stack(first_steps, dim=2).bool()

        horizon_to_index = {h: i for i, h in enumerate(self.horizons)}
        weighted_loss = zero
        weight_total = zero
        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=actions.device
        )

        for step in range(1, self.max_horizon + 1):
            # Native single-stream Dreamer prior transition.  No S5A objective is
            # applied; only the Proto CE below supervises this predicted future.
            current = dynamics.img_step(
                current, action_prefix[:, step - 1], sample=True
            )
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])
            if step not in horizon_to_index:
                continue

            target_indices = starts + step
            target_labels = labels[:, target_indices].reshape(-1).clone()
            valid_mask = valid_prefix.reshape(-1)
            target_labels[~valid_mask] = -1

            pred_feat = self._feature(dynamics, current)
            pred_proj = self._project_feature(pred_feat)
            horizon_loss, stats = self.prototype_bank.loss(pred_proj, target_labels)
            weight = self.horizon_weights[horizon_to_index[step]].to(
                device=horizon_loss.device, dtype=horizon_loss.dtype
            )
            weighted_loss = weighted_loss + weight * horizon_loss
            weight_total = weight_total + weight

            metrics[f"proto_loss_h{step}"] = horizon_loss.detach()
            metrics[f"proto_acc_h{step}"] = stats["acc"].detach()
            metrics[f"proto_num_h{step}"] = stats["num"].detach()

            # Detached diagnostic only: how close the native Dreamer prior is to
            # the replay posterior at the same future horizon.  It is NOT a loss.
            with torch.no_grad():
                target_state = self._flatten_batch_starts(
                    self._gather_state(posterior, target_indices)
                )
                target_proj = self._project_feature(
                    self._feature(dynamics, target_state)
                )
                cosine = (pred_proj.detach() * target_proj).sum(dim=-1)
                mask_f = valid_mask.to(cosine.dtype)
                denom = mask_f.sum().clamp_min(1.0)
                metrics[f"proto_future_cosine_h{step}"] = (
                    (cosine * mask_f).sum() / denom
                ).detach()

        total = weighted_loss / weight_total.clamp_min(self.eps)
        metrics["proto_loss"] = total.detach()
        metrics["proto_starts"] = torch.tensor(
            float(num_starts), device=actions.device
        )
        metrics["proto_enabled"] = torch.tensor(1.0, device=actions.device)
        metrics.update(
            self._deterministic_action_diagnostics(
                dynamics=dynamics,
                posterior=posterior,
                start_state=start_state_flat,
                action_prefix=action_prefix,
                starts=starts,
                first_prefix=first_prefix,
                batch=batch,
                num_starts=num_starts,
            )
        )
        return total, metrics
