"""S-only open-loop multi-step RSSM consistency for ISO3.

Adapted from the user's successful DreamerV3 Experiment 5A. The objective is
restricted to the controllable S branch: starting from replay posterior S_t,
the existing S prior transition is rolled forward with replay action prefixes
and compared to future posterior S states. Z is neither an input to the loss nor
updated by the auxiliary rollout.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
        diagnostics_enabled: bool = True,
        counterfactual_diagnostics: bool = True,
        direct_vs_composed_diagnostics: bool = True,
        direct_vs_composed_horizon: Optional[int] = None,
        direct_vs_composed_midpoint: Optional[int] = None,
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
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.counterfactual_diagnostics = bool(counterfactual_diagnostics)
        self.direct_vs_composed_diagnostics = bool(direct_vs_composed_diagnostics)

        dvc_horizon = (
            self.max_horizon
            if direct_vs_composed_horizon is None
            else int(direct_vs_composed_horizon)
        )
        if dvc_horizon <= 1 or dvc_horizon > self.max_horizon:
            raise ValueError(
                f"direct_vs_composed_horizon={dvc_horizon} must be in "
                f"[2, {self.max_horizon}]"
            )
        dvc_midpoint = (
            max(1, dvc_horizon // 2)
            if direct_vs_composed_midpoint is None
            else int(direct_vs_composed_midpoint)
        )
        if dvc_midpoint <= 0 or dvc_midpoint >= dvc_horizon:
            raise ValueError(
                f"direct_vs_composed_midpoint={dvc_midpoint} must be in "
                f"[1, {dvc_horizon - 1}]"
            )
        self.direct_vs_composed_horizon = dvc_horizon
        self.direct_vs_composed_midpoint = dvc_midpoint

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

    @staticmethod
    def _detach_state(state: TensorDict) -> Dict[str, torch.Tensor]:
        return {key: value.detach() for key, value in state.items()}

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _project_state(
        self, dynamics: nn.Module, state: TensorDict
    ) -> torch.Tensor:
        feat = self._s_feature(dynamics, state)
        raw = self.projector(feat.float())
        return F.normalize(raw, dim=-1, eps=self.eps)

    def _mode_rollout(
        self,
        dynamics: nn.Module,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        capture_steps: Sequence[int],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Deterministic S rollout used only by diagnostics.

        ``sample=False`` is deliberate: diagnostics must not consume the global
        torch RNG, otherwise simply enabling logging would change subsequent
        training samples and break strict ablation comparability.
        """
        wanted = set(int(step) for step in capture_steps)
        current = self._detach_state(start_state)
        captured: Dict[int, Dict[str, torch.Tensor]] = {}
        max_step = max(wanted) if wanted else 0
        for step in range(1, max_step + 1):
            current = dynamics.img_step_s(
                current, action_prefix[:, step - 1], sample=False
            )
            if step in wanted:
                captured[step] = self._detach_state(current)
        return captured

    def _counterfactual_action_diagnostics(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        starts: torch.Tensor,
        first_prefix: torch.Tensor,
        batch: int,
        num_starts: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        """Measure whether S predictions actually respond to the action prefix.

        Three deterministic rollouts start from exactly the same posterior S_t:
          * true: replay action prefix;
          * shuffle: a deterministic cyclic permutation of prefixes across
            batch/start samples (valid one-hot actions, wrong for this state);
          * zero: an all-zero action vector at every step (OOD reference only).

        No diagnostic is added to the training objective.  The cyclic shuffle
        avoids torch.randperm so logging does not consume the global RNG.
        """
        horizons = self.horizons
        true_states = self._mode_rollout(
            dynamics, start_state, action_prefix.detach(), horizons
        )

        if action_prefix.shape[0] > 1:
            shuffled_actions = torch.roll(action_prefix.detach(), shifts=1, dims=0)
        else:
            # Degenerate fallback; normal training uses B*starts >> 1.
            shuffled_actions = torch.flip(action_prefix.detach(), dims=[1])
        zero_actions = torch.zeros_like(action_prefix)

        shuffled_states = self._mode_rollout(
            dynamics, start_state, shuffled_actions, horizons
        )
        zero_states = self._mode_rollout(
            dynamics, start_state, zero_actions, horizons
        )

        metrics: Dict[str, torch.Tensor] = {}
        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=action_prefix.device
        )

        for step in range(1, self.max_horizon + 1):
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])
            if step not in true_states:
                continue

            target_indices = starts + step
            target_state = self._flatten_batch_starts(
                self._gather_s_state(posterior, target_indices)
            )
            target_proj = self._project_state(
                dynamics, self._detach_state(target_state)
            )
            true_proj = self._project_state(dynamics, true_states[step])
            shuffled_proj = self._project_state(dynamics, shuffled_states[step])
            zero_proj = self._project_state(dynamics, zero_states[step])

            true_target_cos = torch.sum(true_proj * target_proj, dim=-1)
            shuffled_target_cos = torch.sum(shuffled_proj * target_proj, dim=-1)
            zero_target_cos = torch.sum(zero_proj * target_proj, dim=-1)
            true_shuffle_cos = torch.sum(true_proj * shuffled_proj, dim=-1)
            true_zero_cos = torch.sum(true_proj * zero_proj, dim=-1)

            mask = valid_prefix.reshape(-1)
            true_target = self._masked_mean(true_target_cos, mask)
            shuffle_target = self._masked_mean(shuffled_target_cos, mask)
            zero_target = self._masked_mean(zero_target_cos, mask)

            metrics[f"s_cf_true_target_cos_h{step}"] = true_target
            metrics[f"s_cf_shuffle_target_cos_h{step}"] = shuffle_target
            metrics[f"s_cf_zero_target_cos_h{step}"] = zero_target
            # Positive margin: the correct replay prefix predicts the actual
            # future S better than the counterfactual prefix.
            metrics[f"s_cf_margin_shuffle_h{step}"] = (
                true_target - shuffle_target
            )
            metrics[f"s_cf_margin_zero_h{step}"] = true_target - zero_target
            # Response gap measures sensitivity only; a large gap is not by
            # itself evidence that the action response is correct.
            metrics[f"s_cf_response_gap_shuffle_h{step}"] = 1.0 - self._masked_mean(
                true_shuffle_cos, mask
            )
            metrics[f"s_cf_response_gap_zero_h{step}"] = 1.0 - self._masked_mean(
                true_zero_cos, mask
            )
            metrics[f"s_cf_valid_h{step}"] = mask.float().mean()

        return metrics, true_states

    def _direct_vs_composed_diagnostics(
        self,
        dynamics: nn.Module,
        posterior: TensorDict,
        start_state: TensorDict,
        action_prefix: torch.Tensor,
        starts: torch.Tensor,
        first_prefix: torch.Tensor,
        batch: int,
        num_starts: int,
        true_mode_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Open-loop vs posterior-reanchored terminal consistency diagnostic.

        Fast-LeWM can compare a *direct prefix predictor* with a composed
        prefix prediction.  This ISO3 S5A has only one autoregressive one-step
        RSSM transition, so splitting the same predicted rollout in two would
        be algebraically identical and therefore useless as a diagnostic.

        We use the closest informative surrogate for this architecture:
          direct: S_t --true actions--> S_hat_{t+h} fully open-loop;
          composed/reanchored: take the replay posterior S_{t+m} as an
            intermediate anchor, then roll the remaining action suffix to h.

        High direct-vs-reanchored consistency means the terminal estimate is
        insensitive to a real midpoint correction.  A large reanchor gain
        means early open-loop drift is materially hurting the terminal state.
        This is diagnostic/confidence only and never enters the loss.
        """
        horizon = self.direct_vs_composed_horizon
        midpoint = self.direct_vs_composed_midpoint

        if true_mode_states is not None and horizon in true_mode_states:
            direct_state = true_mode_states[horizon]
        else:
            direct_state = self._mode_rollout(
                dynamics, start_state, action_prefix.detach(), [horizon]
            )[horizon]

        midpoint_indices = starts + midpoint
        composed_state = self._flatten_batch_starts(
            self._gather_s_state(posterior, midpoint_indices)
        )
        composed_state = self._detach_state(composed_state)
        for step in range(midpoint + 1, horizon + 1):
            composed_state = dynamics.img_step_s(
                composed_state, action_prefix[:, step - 1], sample=False
            )
        composed_state = self._detach_state(composed_state)

        target_indices = starts + horizon
        target_state = self._flatten_batch_starts(
            self._gather_s_state(posterior, target_indices)
        )
        target_state = self._detach_state(target_state)

        direct_proj = self._project_state(dynamics, direct_state)
        composed_proj = self._project_state(dynamics, composed_state)
        target_proj = self._project_state(dynamics, target_state)

        direct_composed_cos = torch.sum(direct_proj * composed_proj, dim=-1)
        direct_target_cos = torch.sum(direct_proj * target_proj, dim=-1)
        composed_target_cos = torch.sum(composed_proj * target_proj, dim=-1)

        valid_prefix = torch.ones(
            batch, num_starts, dtype=torch.bool, device=action_prefix.device
        )
        for step in range(1, horizon + 1):
            valid_prefix = valid_prefix & (~first_prefix[:, :, step - 1])
        mask = valid_prefix.reshape(-1)

        consistency_cos = self._masked_mean(direct_composed_cos, mask)
        direct_target = self._masked_mean(direct_target_cos, mask)
        composed_target = self._masked_mean(composed_target_cos, mask)

        return {
            f"s_dvc_reanchored_consistency_cos_h{horizon}": consistency_cos,
            f"s_dvc_reanchored_gap_h{horizon}": 1.0 - consistency_cos,
            # Convenience confidence in [0,1] derived from cosine [-1,1].
            f"s_dvc_reanchored_confidence_h{horizon}": torch.clamp(
                0.5 * (consistency_cos + 1.0), 0.0, 1.0
            ),
            f"s_dvc_direct_target_cos_h{horizon}": direct_target,
            f"s_dvc_reanchored_target_cos_h{horizon}": composed_target,
            f"s_dvc_reanchor_gain_h{horizon}": composed_target - direct_target,
            f"s_dvc_valid_h{horizon}": mask.float().mean(),
            "s_dvc_horizon": torch.tensor(
                float(horizon), device=action_prefix.device
            ),
            "s_dvc_midpoint": torch.tensor(
                float(midpoint), device=action_prefix.device
            ),
        }

    def _run_diagnostics(
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
        if not self.diagnostics_enabled:
            return {}

        metrics: Dict[str, torch.Tensor] = {}
        true_mode_states = None
        with torch.no_grad():
            if self.counterfactual_diagnostics:
                cf_metrics, true_mode_states = self._counterfactual_action_diagnostics(
                    dynamics,
                    posterior,
                    start_state,
                    action_prefix,
                    starts,
                    first_prefix,
                    batch,
                    num_starts,
                )
                metrics.update(cf_metrics)

            if self.direct_vs_composed_diagnostics:
                metrics.update(
                    self._direct_vs_composed_diagnostics(
                        dynamics,
                        posterior,
                        start_state,
                        action_prefix,
                        starts,
                        first_prefix,
                        batch,
                        num_starts,
                        true_mode_states=true_mode_states,
                    )
                )

        # Guarantee that diagnostic values are detached even if implementation
        # changes later.  They are never added to the optimization objective.
        return {name: value.detach() for name, value in metrics.items()}

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
        start_state_flat = self._flatten_batch_starts(start_state)
        current = start_state_flat

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

        # Diagnostics are deliberately outside the optimization objective.
        # They use deterministic mode rollouts and torch.no_grad(), so enabling
        # them does not consume RNG or add any gradient path to S/Z.
        metrics.update(
            self._run_diagnostics(
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
