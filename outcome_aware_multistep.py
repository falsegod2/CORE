"""Outcome-aware multi-step latent dynamics for Experiment 5D.

The objective acts directly on the standard DreamerV3 RSSM open-loop rollout.
It does not add a second latent branch, object slots, a planner, intrinsic reward,
or a separate action-prefix Transformer.

For posterior start states sampled from replay, the existing RSSM is rolled out
with replay actions. The objective combines:

1. weak absolute future-state consistency;
2. action-induced latent-change consistency;
3. open-loop cumulative reward consistency using the existing reward head;
4. pairwise return ranking;
5. an optional real-action versus shuffled-action margin.

All target posterior features and replay outcomes are stop-gradient targets.
Episode boundaries are masked. Curriculum gates are functions of the actual
environment step supplied by the agent, not optimizer update count.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from multistep_consistency import FixedRandomProjector


TensorDict = Mapping[str, torch.Tensor]


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


class OutcomeAwareMultiStepRSSM(nn.Module):
    """Result-aware regularization of the existing RSSM imagination path.

    The class intentionally has no trainable latent predictor. The only module
    it owns is a frozen random projector. Reward predictions are produced by the
    original Dreamer reward head, so outcome gradients must pass through the
    same RSSM states that the actor later uses for imagination.
    """

    def __init__(
        self,
        feat_dim: int,
        horizons: Sequence[int] = (1, 2, 4, 8, 15),
        horizon_weights: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.0),
        projection_dim: int = 512,
        starts_per_sequence: int = 4,
        projection_seed: int = 314159,
        discount: float = 0.997,
        absolute_loss_scale: float = 0.005,
        delta_loss_scale: float = 0.020,
        return_loss_scale: float = 0.010,
        return_rank_loss_scale: float = 0.005,
        action_margin_loss_scale: float = 0.005,
        action_margin: float = 0.10,
        action_negative_stop_gradient: bool = True,
        rollout_sample: bool = False,
        preserve_rng_state: bool = False,
        detach_start_state: bool = True,
        delta_min_norm: float = 1e-4,
        return_huber_delta: float = 1.0,
        return_weight_bonus: float = 1.0,
        return_weight_cap: float = 4.0,
        ranking_epsilon: float = 1e-4,
        ranking_temperature: float = 1.0,
        ranking_std_floor: float = 0.05,
        global_start_step: int = 50_000,
        global_ramp_steps: int = 250_000,
        horizon_start_steps: Sequence[int] = (50_000, 50_000, 50_000, 150_000, 300_000),
        horizon_ramp_steps: Sequence[int] = (100_000, 100_000, 100_000, 100_000, 100_000),
        compute_action_gap: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        horizons = tuple(int(x) for x in horizons)
        weights = tuple(float(x) for x in horizon_weights)
        starts = tuple(int(x) for x in horizon_start_steps)
        ramps = tuple(int(x) for x in horizon_ramp_steps)

        if not horizons or any(x <= 0 for x in horizons):
            raise ValueError(f"Invalid horizons: {horizons}")
        if sorted(set(horizons)) != list(horizons):
            raise ValueError(f"Horizons must be unique and sorted: {horizons}")
        if not (len(horizons) == len(weights) == len(starts) == len(ramps)):
            raise ValueError((horizons, weights, starts, ramps))
        if any(x < 0.0 for x in weights) or sum(weights) <= 0.0:
            raise ValueError(f"Invalid horizon weights: {weights}")
        if any(x < 0 for x in starts) or any(x < 0 for x in ramps):
            raise ValueError((starts, ramps))
        if starts_per_sequence <= 0:
            raise ValueError(starts_per_sequence)
        if not 0.0 < discount <= 1.0:
            raise ValueError(discount)
        if action_margin < 0.0:
            raise ValueError(action_margin)
        if return_huber_delta <= 0.0:
            raise ValueError(return_huber_delta)
        if return_weight_bonus < 0.0 or return_weight_cap < 0.0:
            raise ValueError((return_weight_bonus, return_weight_cap))
        if ranking_epsilon < 0.0 or ranking_temperature <= 0.0:
            raise ValueError((ranking_epsilon, ranking_temperature))
        if ranking_std_floor <= 0.0:
            raise ValueError(ranking_std_floor)

        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.starts_per_sequence = int(starts_per_sequence)
        self.discount = float(discount)
        self.absolute_loss_scale = float(absolute_loss_scale)
        self.delta_loss_scale = float(delta_loss_scale)
        self.return_loss_scale = float(return_loss_scale)
        self.return_rank_loss_scale = float(return_rank_loss_scale)
        self.action_margin_loss_scale = float(action_margin_loss_scale)
        self.action_margin = float(action_margin)
        self.action_negative_stop_gradient = bool(action_negative_stop_gradient)
        self.rollout_sample = bool(rollout_sample)
        self.preserve_rng_state = bool(preserve_rng_state)
        self.detach_start_state = bool(detach_start_state)
        self.delta_min_norm = float(delta_min_norm)
        self.return_huber_delta = float(return_huber_delta)
        self.return_weight_bonus = float(return_weight_bonus)
        self.return_weight_cap = float(return_weight_cap)
        self.ranking_epsilon = float(ranking_epsilon)
        self.ranking_temperature = float(ranking_temperature)
        self.ranking_std_floor = float(ranking_std_floor)
        self.global_start_step = int(global_start_step)
        self.global_ramp_steps = int(global_ramp_steps)
        self.compute_action_gap = bool(compute_action_gap)
        self.eps = float(eps)

        self.projector = FixedRandomProjector(
            int(feat_dim), int(projection_dim), int(projection_seed)
        )
        self.register_buffer(
            "horizon_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "horizon_start_steps",
            torch.tensor(starts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "horizon_ramp_steps",
            torch.tensor(ramps, dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _state_feature(dynamics: nn.Module, state: TensorDict) -> torch.Tensor:
        """Smooth RSSM feature: distribution probabilities/means plus deter."""
        if "deter" in state:
            deter = state["deter"]
            if getattr(dynamics, "_discrete", False):
                stoch = torch.softmax(state["logit"], dim=-1)
                stoch = stoch.reshape(*stoch.shape[:-2], -1)
            else:
                stoch = state["mean"]
            return torch.cat([stoch, deter], dim=-1)
        raise KeyError(
            "OutcomeAwareMultiStepRSSM requires the standard unified RSSM; "
            f"received state keys {tuple(state.keys())}."
        )

    @staticmethod
    def _gather_state(state: TensorDict, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {key: value[:, indices] for key, value in state.items()}

    @staticmethod
    def _flatten_batch_starts(state: TensorDict) -> Dict[str, torch.Tensor]:
        return {
            key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            for key, value in state.items()
        }

    @staticmethod
    def _detach_state(state: TensorDict) -> Dict[str, torch.Tensor]:
        return {key: value.detach() for key, value in state.items()}

    @staticmethod
    def _as_scalar_sequence(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim != 2:
            raise ValueError(f"Expected {name} [B,T] or [B,T,1], got {x.shape}")
        return x.float()

    def _start_indices(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        usable = sequence_length - self.max_horizon
        if usable <= 0:
            raise ValueError(
                f"batch_length={sequence_length} must exceed "
                f"max_horizon={self.max_horizon}."
            )
        count = min(self.starts_per_sequence, usable)
        if count == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(0, usable - 1, steps=count, device=device).round().long()

    def _ramp(
        self,
        step: torch.Tensor,
        start: torch.Tensor | float,
        duration: torch.Tensor | float,
    ) -> torch.Tensor:
        start_t = torch.as_tensor(start, device=step.device, dtype=step.dtype)
        duration_t = torch.as_tensor(duration, device=step.device, dtype=step.dtype)
        safe_duration = duration_t.clamp_min(1.0)
        linear = ((step - start_t) / safe_duration).clamp(0.0, 1.0)
        hard = (step >= start_t).to(step.dtype)
        return torch.where(duration_t > 0.0, linear, hard)

    def _global_gate(self, env_step: torch.Tensor) -> torch.Tensor:
        return self._ramp(env_step, float(self.global_start_step), float(self.global_ramp_steps))

    def _horizon_gate(self, env_step: torch.Tensor, index: int) -> torch.Tensor:
        starts = self.horizon_start_steps.to(device=env_step.device, dtype=env_step.dtype)
        ramps = self.horizon_ramp_steps.to(device=env_step.device, dtype=env_step.dtype)
        return self._ramp(env_step, starts[index], ramps[index])

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def _masked_mean_std(
        self, value: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._masked_mean(value, mask)
        var = self._masked_mean((value - mean).square(), mask)
        return mean, torch.sqrt(var + self.eps)

    def _masked_pearson(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x_mean, x_std = self._masked_mean_std(x, mask)
        y_mean, y_std = self._masked_mean_std(y, mask)
        covariance = self._masked_mean((x - x_mean) * (y - y_mean), mask)
        return covariance / (x_std * y_std + self.eps)

    def _weighted_horizon_average(
        self,
        sums: Dict[int, torch.Tensor],
        weights: Dict[int, torch.Tensor],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        numerator = zero
        denominator = zero
        for horizon in self.horizons:
            numerator = numerator + sums[horizon]
            denominator = denominator + weights[horizon]
        return numerator / denominator.clamp_min(self.eps)

    def forward(
        self,
        dynamics: nn.Module,
        reward_head: nn.Module,
        posterior: TensorDict,
        actions: torch.Tensor,
        is_first: torch.Tensor,
        rewards: torch.Tensor,
        ends: torch.Tensor,
        env_step: torch.Tensor | float | int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,T,A], got {actions.shape}")
        batch, time, action_dim = actions.shape
        rewards = self._as_scalar_sequence(rewards, "rewards")
        ends = self._as_scalar_sequence(ends, "ends").clamp(0.0, 1.0)
        if rewards.shape != (batch, time) or ends.shape != (batch, time):
            raise ValueError((actions.shape, rewards.shape, ends.shape))
        if is_first.ndim == 3 and is_first.shape[-1] == 1:
            is_first = is_first[..., 0]
        if is_first.shape != (batch, time):
            raise ValueError((actions.shape, is_first.shape))

        zero = actions.sum() * 0.0
        if time <= self.max_horizon:
            return zero, {
                "outcome_aware_total_loss": zero.detach(),
                "outcome_aware_valid_fraction": zero.detach(),
            }

        if env_step is None:
            env_step_t = torch.zeros((), device=actions.device, dtype=torch.float32)
        else:
            env_step_t = torch.as_tensor(
                env_step, device=actions.device, dtype=torch.float32
            ).reshape(())

        starts = self._start_indices(time, actions.device)
        num_starts = int(starts.numel())
        flat_count = batch * num_starts

        start_state = self._flatten_batch_starts(self._gather_state(posterior, starts))
        current_real = self._detach_state(start_state) if self.detach_start_state else dict(start_state)
        current_shuffle = self._detach_state(start_state) if self.detach_start_state else {
            key: value.clone() for key, value in start_state.items()
        }

        action_steps = []
        first_steps = []
        for offset in range(1, self.max_horizon + 1):
            action_steps.append(actions[:, starts + offset])
            first_steps.append(is_first[:, starts + offset])
        action_prefix = torch.stack(action_steps, dim=2)  # [B,S,H,A]
        first_prefix = torch.stack(first_steps, dim=2).bool()  # [B,S,H]
        action_prefix = action_prefix.reshape(flat_count, self.max_horizon, action_dim)
        first_prefix_flat = first_prefix.reshape(flat_count, self.max_horizon)

        # Deterministic cyclic permutation: no global RNG consumption and no
        # accidental changes to seed-matched baseline initialization/training.
        permutation = torch.roll(
            torch.arange(flat_count, device=actions.device), shifts=1, dims=0
        )
        shuffled_actions = action_prefix[permutation]
        shuffled_first = first_prefix_flat[permutation]
        compute_negative = (
            (self.compute_action_gap or self.action_margin_loss_scale > 0.0)
            and flat_count > 1
        )

        with torch.no_grad():
            start_feature = self._state_feature(dynamics, start_state)
            start_projection = self.projector(start_feature.float())

        valid_real = torch.ones(flat_count, dtype=torch.bool, device=actions.device)
        valid_shuffle = torch.ones_like(valid_real)
        alive = torch.ones(flat_count, dtype=torch.float32, device=actions.device)
        target_return = torch.zeros_like(alive)
        predicted_return = torch.zeros_like(alive)

        horizon_to_index = {h: i for i, h in enumerate(self.horizons)}
        component_sums = {
            name: {h: zero for h in self.horizons}
            for name in ("absolute", "delta", "return", "rank", "action")
        }
        component_weights = {
            name: {h: zero for h in self.horizons}
            for name in ("absolute", "delta", "return", "rank", "action")
        }
        metrics: Dict[str, torch.Tensor] = {}

        future_offsets = torch.arange(
            1, self.max_horizon + 1, device=actions.device
        )
        future_indices = starts[:, None] + future_offsets[None, :]  # [S,H]
        reward_matrix = rewards[:, future_indices].reshape(
            flat_count, self.max_horizon
        )
        end_matrix = ends[:, future_indices].reshape(
            flat_count, self.max_horizon
        )

        cpu_rng_state = None
        cuda_rng_state = None
        cuda_device = None
        if self.rollout_sample and self.preserve_rng_state:
            cpu_rng_state = torch.random.get_rng_state()
            if actions.device.type == "cuda":
                cuda_device = actions.device.index
                if cuda_device is None:
                    cuda_device = torch.cuda.current_device()
                cuda_rng_state = torch.cuda.get_rng_state(cuda_device)

        for step in range(1, self.max_horizon + 1):
            current_real = dynamics.img_step(
                current_real,
                action_prefix[:, step - 1],
                sample=self.rollout_sample,
            )
            if compute_negative:
                if self.action_negative_stop_gradient:
                    with torch.no_grad():
                        current_shuffle = dynamics.img_step(
                            current_shuffle,
                            shuffled_actions[:, step - 1],
                            sample=self.rollout_sample,
                        )
                else:
                    current_shuffle = dynamics.img_step(
                        current_shuffle,
                        shuffled_actions[:, step - 1],
                        sample=self.rollout_sample,
                    )

            valid_real = valid_real & (~first_prefix_flat[:, step - 1])
            valid_shuffle = valid_shuffle & (~shuffled_first[:, step - 1])

            reward_step = reward_matrix[:, step - 1]
            end_step = end_matrix[:, step - 1]
            discount = self.discount ** (step - 1)
            target_return = target_return + discount * alive * reward_step

            reward_dist = reward_head(dynamics.get_feat(current_real))
            predicted_reward = reward_dist.mode()
            if predicted_reward.ndim > 1:
                predicted_reward = predicted_reward.squeeze(-1)
            if predicted_reward.shape != alive.shape:
                raise ValueError(
                    f"Reward head mode must be [B*S] or [B*S,1], got "
                    f"{predicted_reward.shape}."
                )
            predicted_return = predicted_return + discount * alive * predicted_reward.float()
            alive = alive * (1.0 - end_step)

            if step not in horizon_to_index:
                continue

            index = horizon_to_index[step]
            base_weight = self.horizon_weights[index].to(
                device=actions.device, dtype=torch.float32
            )
            horizon_gate = self._horizon_gate(env_step_t, index)

            target_state = self._flatten_batch_starts(
                self._gather_state(posterior, starts + step)
            )
            predicted_feature = self._state_feature(dynamics, current_real)
            predicted_projection = self.projector(predicted_feature.float())
            with torch.no_grad():
                target_feature = self._state_feature(dynamics, target_state)
                target_projection = self.projector(target_feature.float())

            predicted_unit = F.normalize(predicted_projection, dim=-1, eps=self.eps)
            target_unit = F.normalize(target_projection, dim=-1, eps=self.eps)
            absolute_cosine = torch.sum(predicted_unit * target_unit, dim=-1)
            absolute_vector = 1.0 - absolute_cosine
            absolute_mask = valid_real
            absolute_count = absolute_mask.float().sum()
            absolute_loss = self._masked_mean(absolute_vector, absolute_mask)
            absolute_mean_cos = self._masked_mean(absolute_cosine, absolute_mask)

            target_delta = target_projection - start_projection
            predicted_delta = predicted_projection - start_projection
            target_delta_norm = target_delta.norm(dim=-1)
            predicted_delta_norm = predicted_delta.norm(dim=-1)
            delta_mask = valid_real & (target_delta_norm > self.delta_min_norm)
            target_delta_unit = F.normalize(target_delta, dim=-1, eps=self.eps)
            predicted_delta_unit = F.normalize(predicted_delta, dim=-1, eps=self.eps)
            delta_cosine = torch.sum(predicted_delta_unit * target_delta_unit, dim=-1)
            delta_vector = 1.0 - delta_cosine
            delta_count = delta_mask.float().sum()
            delta_loss = self._masked_mean(delta_vector, delta_mask)
            delta_mean_cos = self._masked_mean(delta_cosine, delta_mask)

            valid_float = valid_real.float()
            return_target = target_return.detach()
            return_prediction = predicted_return
            return_target_symlog = symlog(return_target)
            return_prediction_symlog = symlog(return_prediction)
            return_mean, return_std = self._masked_mean_std(
                return_target_symlog, valid_real
            )
            standardized_advantage = (
                (return_target_symlog - return_mean) / return_std.clamp_min(self.eps)
            ).clamp_min(0.0)
            sample_weight = 1.0 + self.return_weight_bonus * standardized_advantage.clamp(
                max=self.return_weight_cap
            )
            return_vector = F.smooth_l1_loss(
                return_prediction_symlog,
                return_target_symlog,
                reduction="none",
                beta=self.return_huber_delta,
            )
            weighted_return_mask = valid_float * sample_weight.detach()
            return_loss = (
                return_vector * weighted_return_mask
            ).sum() / weighted_return_mask.sum().clamp_min(1.0)
            return_mae = self._masked_mean(
                torch.abs(return_prediction - return_target), valid_real
            )
            return_pearson = self._masked_pearson(
                return_prediction_symlog,
                return_target_symlog,
                valid_real,
            )

            pair_target = return_target_symlog[permutation]
            pair_prediction = return_prediction_symlog[permutation]
            target_difference = return_target_symlog - pair_target
            prediction_difference = return_prediction_symlog - pair_prediction
            pair_valid = valid_real & valid_real[permutation]
            informative_pair = pair_valid & (
                torch.abs(target_difference) > self.ranking_epsilon
            )
            target_sign = torch.sign(target_difference).detach()
            rank_accuracy = self._masked_mean(
                (torch.sign(prediction_difference) == target_sign).float(),
                informative_pair,
            )
            rank_count = informative_pair.float().sum()
            rank_loss = zero
            if self.return_rank_loss_scale > 0.0:
                normalized_prediction_difference = prediction_difference / return_std.detach().clamp_min(
                    self.ranking_std_floor
                )
                ranking_vector = F.softplus(
                    -target_sign
                    * normalized_prediction_difference
                    / self.ranking_temperature
                )
                rank_loss = self._masked_mean(ranking_vector, informative_pair)

            action_loss = zero
            action_gap = zero.detach()
            shuffled_cosine_mean = zero.detach()
            action_count = zero.detach()
            if compute_negative:
                shuffled_feature = self._state_feature(dynamics, current_shuffle)
                shuffled_projection = self.projector(shuffled_feature.float())
                shuffled_delta = shuffled_projection - start_projection
                shuffled_delta_unit = F.normalize(
                    shuffled_delta, dim=-1, eps=self.eps
                )
                shuffled_cosine = torch.sum(
                    shuffled_delta_unit * target_delta_unit, dim=-1
                )
                action_mask = delta_mask & valid_shuffle
                if self.action_margin_loss_scale > 0.0:
                    negative = (
                        shuffled_cosine.detach()
                        if self.action_negative_stop_gradient
                        else shuffled_cosine
                    )
                    action_vector = F.relu(
                        self.action_margin - delta_cosine + negative
                    )
                    action_loss = self._masked_mean(action_vector, action_mask)
                action_gap = self._masked_mean(
                    delta_cosine - shuffled_cosine, action_mask
                ).detach()
                shuffled_cosine_mean = self._masked_mean(
                    shuffled_cosine, action_mask
                ).detach()
                action_count = action_mask.float().sum().detach()

            has_absolute = (absolute_count > 0).float()
            has_delta = (delta_count > 0).float()
            has_return = (valid_float.sum() > 0).float()
            has_rank = (rank_count > 0).float()
            has_action = (action_count > 0).float()

            effective_base = base_weight * horizon_gate
            for name, value, available in (
                ("absolute", absolute_loss, has_absolute),
                ("delta", delta_loss, has_delta),
                ("return", return_loss, has_return),
                ("rank", rank_loss, has_rank),
                ("action", action_loss, has_action),
            ):
                effective_weight = effective_base * available
                component_sums[name][step] = effective_weight * value
                component_weights[name][step] = effective_weight

            metrics[f"oa_abs_loss_h{step}"] = absolute_loss.detach()
            metrics[f"oa_abs_cosine_h{step}"] = absolute_mean_cos.detach()
            metrics[f"oa_delta_loss_h{step}"] = delta_loss.detach()
            metrics[f"oa_delta_cosine_h{step}"] = delta_mean_cos.detach()
            metrics[f"oa_delta_target_norm_h{step}"] = self._masked_mean(
                target_delta_norm, delta_mask
            ).detach()
            metrics[f"oa_delta_pred_norm_h{step}"] = self._masked_mean(
                predicted_delta_norm, delta_mask
            ).detach()
            metrics[f"oa_return_loss_h{step}"] = return_loss.detach()
            metrics[f"oa_return_mae_h{step}"] = return_mae.detach()
            metrics[f"oa_return_pearson_h{step}"] = return_pearson.detach()
            metrics[f"oa_return_target_mean_h{step}"] = self._masked_mean(
                return_target, valid_real
            ).detach()
            metrics[f"oa_return_pred_mean_h{step}"] = self._masked_mean(
                return_prediction, valid_real
            ).detach()
            metrics[f"oa_return_sample_weight_h{step}"] = self._masked_mean(
                sample_weight, valid_real
            ).detach()
            metrics[f"oa_rank_loss_h{step}"] = rank_loss.detach()
            metrics[f"oa_rank_accuracy_h{step}"] = rank_accuracy.detach()
            metrics[f"oa_rank_pair_fraction_h{step}"] = informative_pair.float().mean().detach()
            metrics[f"oa_action_margin_loss_h{step}"] = action_loss.detach()
            metrics[f"oa_action_gap_h{step}"] = action_gap
            metrics[f"oa_shuffled_delta_cosine_h{step}"] = shuffled_cosine_mean
            metrics[f"oa_valid_h{step}"] = valid_real.float().mean().detach()
            metrics[f"oa_delta_valid_h{step}"] = delta_mask.float().mean().detach()
            metrics[f"oa_horizon_gate_h{step}"] = horizon_gate.detach()

        if cpu_rng_state is not None:
            torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None and cuda_device is not None:
            torch.cuda.set_rng_state(cuda_rng_state, cuda_device)

        raw_absolute = self._weighted_horizon_average(
            component_sums["absolute"], component_weights["absolute"], zero
        )
        raw_delta = self._weighted_horizon_average(
            component_sums["delta"], component_weights["delta"], zero
        )
        raw_return = self._weighted_horizon_average(
            component_sums["return"], component_weights["return"], zero
        )
        raw_rank = self._weighted_horizon_average(
            component_sums["rank"], component_weights["rank"], zero
        )
        raw_action = self._weighted_horizon_average(
            component_sums["action"], component_weights["action"], zero
        )

        global_gate = self._global_gate(env_step_t)
        unscaled_total = (
            self.absolute_loss_scale * raw_absolute
            + self.delta_loss_scale * raw_delta
            + self.return_loss_scale * raw_return
            + self.return_rank_loss_scale * raw_rank
            + self.action_margin_loss_scale * raw_action
        )
        total = global_gate * unscaled_total

        metrics["oa_absolute_loss"] = raw_absolute.detach()
        metrics["oa_delta_loss"] = raw_delta.detach()
        metrics["oa_return_loss"] = raw_return.detach()
        metrics["oa_rank_loss"] = raw_rank.detach()
        metrics["oa_action_margin_loss"] = raw_action.detach()
        metrics["oa_absolute_contribution"] = (
            global_gate * self.absolute_loss_scale * raw_absolute
        ).detach()
        metrics["oa_delta_contribution"] = (
            global_gate * self.delta_loss_scale * raw_delta
        ).detach()
        metrics["oa_return_contribution"] = (
            global_gate * self.return_loss_scale * raw_return
        ).detach()
        metrics["oa_rank_contribution"] = (
            global_gate * self.return_rank_loss_scale * raw_rank
        ).detach()
        metrics["oa_action_contribution"] = (
            global_gate * self.action_margin_loss_scale * raw_action
        ).detach()
        metrics["outcome_aware_curriculum_scale"] = global_gate.detach()
        metrics["outcome_aware_env_step"] = env_step_t.detach()
        metrics["outcome_aware_total_loss"] = total.detach()
        metrics["outcome_aware_starts"] = torch.tensor(
            float(num_starts), device=actions.device
        )
        metrics["outcome_aware_valid_fraction"] = valid_real.float().mean().detach()
        return total, metrics
