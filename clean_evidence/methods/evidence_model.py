"""Evidence-conditioned Dreamer world model and recorded-prefix objectives.

The RGB observation writer is history independent. Teacher maps are targets
only, and reward consistency reads the same task-reward head as behavior.
This is an experimental architecture, not an established control guarantee.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..vendor.dreamerv3_torch import models, networks, tools
from .evidence_rssm import EvidenceRSSM


DEFAULT_OPTIONS = {
    "memory_observations": 16, "tokens_per_observation": 4,
    "memory_dim": 256, "attention_heads": 4, "memory_enabled": True,
    "read_mode": "each_step", "writer_depth": 32, "heatmap_size": 8,
    "burnin": 32, "local_heat_weight": 1.0, "posterior_heat_weight": 1.0,
    "fc_weight": 0.05, "fc_hidden_weight": 0.01, "fc_heat_weight": 1.0,
    "rc_weight": 0.01, "task_reward_weight": 1.0, "clip_reward_weight": 1.0,
    "clip_scale": 1.0, "horizons": [1, 2, 4, 8, 15],
    "horizon_weights": [1.0, 1.0, 0.75, 0.5, 0.25],
    "starts_per_sequence": 4, "mc_samples": 2,
    "freeze_rc_reward_head": True, "seed": 0,
}


@contextlib.contextmanager
def _freeze_parameters(module):
    """Freeze parameter updates for this forward while retaining input grads."""
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


class RGBEvidenceWriter(nn.Module):
    """Four quadrant summaries computed from the current RGB, never labels."""
    def __init__(self, memory_dim=256, depth=32, map_size=8):
        super().__init__()
        if map_size < 2 or map_size % 2:
            raise ValueError("heatmap_size must be an even integer >= 2")
        self.map_size, self.memory_dim = map_size, memory_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, depth, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(depth, depth * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((map_size, map_size)),
        )
        self.local_heat = nn.Conv2d(depth * 2, 1, 1)
        self.token_projection = nn.Linear(depth * 2, memory_dim)
        self.quadrant_embedding = nn.Parameter(torch.zeros(4, memory_dim))

    def forward(self, image):
        if image.shape[-1] != 3 or image.ndim < 4:
            raise ValueError("Writer expects (...,height,width,3) RGB normalized to [0,1]")
        batch_shape = image.shape[:-3]
        flat = image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2)
        features = self.features(flat)
        logits = self.local_heat(features)[:, 0]
        weights = 0.05 + logits.sigmoid()
        half = self.map_size // 2
        tokens = []
        for row, column in ((0, 0), (0, half), (half, 0), (half, half)):
            region = features[:, :, row:row + half, column:column + half]
            weight = weights[:, None, row:row + half, column:column + half]
            summary = (region * weight).sum((-1, -2)) / weight.sum((-1, -2)).clamp_min(1e-6)
            tokens.append(self.token_projection(summary))
        tokens = torch.stack(tokens, 1) + self.quadrant_embedding
        return (tokens.reshape(*batch_shape, 4, self.memory_dim),
                logits.reshape(*batch_shape, self.map_size, self.map_size))


class EvidenceWorldModel(models.WorldModel):
    """Registered modules, shared transition, masked burn-in and FC/RC."""
    def __init__(self, obs_space, act_space, step, config, evidence_options=None):
        # Build directly instead of allocating a discarded baseline RSSM/opt.
        nn.Module.__init__(self)
        options = dict(DEFAULT_OPTIONS)
        supplied = dict(evidence_options or {})
        unknown = set(supplied) - set(options)
        if unknown:
            raise ValueError(f"Unknown evidence options: {sorted(unknown)}")
        options.update(supplied)
        if options["tokens_per_observation"] != 4:
            raise ValueError("The auditable quadrant writer uses exactly four tokens per observation")
        for name in ("local_heat_weight", "posterior_heat_weight", "fc_weight", "fc_hidden_weight", "fc_heat_weight", "rc_weight", "task_reward_weight", "clip_reward_weight", "clip_scale"):
            if isinstance(options[name], bool) or not isinstance(options[name], (float, int)) or not math.isfinite(float(options[name])) or options[name] < 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar")
        for name in ("memory_enabled", "freeze_rc_reward_head"):
            if not isinstance(options[name], bool):
                raise ValueError(f"{name} must be a boolean")
        if len(options["horizons"]) != len(options["horizon_weights"]) or not options["horizons"]:
            raise ValueError("horizons and horizon_weights must be nonempty and aligned")
        if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in options["horizons"]):
            raise ValueError("horizons must be positive integers")
        if sorted(set(options["horizons"])) != options["horizons"]:
            raise ValueError("horizons must be unique and strictly increasing")
        if any(isinstance(w, bool) or not isinstance(w, (float, int)) or not math.isfinite(float(w)) or float(w) < 0 for w in options["horizon_weights"]) or sum(options["horizon_weights"]) <= 0:
            raise ValueError("horizon_weights must be nonnegative with positive total")
        for name in ("starts_per_sequence", "mc_samples", "writer_depth", "heatmap_size"):
            if isinstance(options[name], bool) or not isinstance(options[name], int) or options[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if options["clip_scale"] and not options["clip_reward_weight"]:
            raise ValueError("An active clip policy route requires a trained clip_reward head")
        if not options["task_reward_weight"]:
            raise ValueError("The task reward policy route requires per-step supervision")
        self.options = options
        self._step, self._config = step, config
        self._use_amp = config.precision == 16
        shapes = {key: tuple(value.shape) for key, value in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = EvidenceRSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act, config.norm,
            config.dyn_mean_act, config.dyn_std_act, config.dyn_min_std,
            config.unimix_ratio, config.initial, config.num_actions,
            self.embed_size, config.device,
            memory_observations=options["memory_observations"],
            memory_dim=options["memory_dim"], attention_heads=options["attention_heads"],
            memory_enabled=options["memory_enabled"], read_mode=options["read_mode"],
            seed=options["seed"],
        )
        feat_size = config.dyn_deter + config.dyn_stoch * config.dyn_discrete
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(feat_size, shapes, **config.decoder)
        def reward_head(name):
            return networks.MLP(
                feat_size, (255,) if config.reward_head["dist"] == "symlog_disc" else (),
                config.reward_head["layers"], config.units, config.act, config.norm,
                dist=config.reward_head["dist"], outscale=config.reward_head["outscale"],
                device=config.device, name=name,
            )
        self.heads["task_reward"] = reward_head("TaskReward")
        if options["clip_reward_weight"] > 0:
            self.heads["clip_reward"] = reward_head("ClipReward")
        self.heads["cont"] = networks.MLP(
            feat_size, (), config.cont_head["layers"], config.units,
            config.act, config.norm, dist="binary", outscale=config.cont_head["outscale"],
            device=config.device, name="Cont",
        )
        needs_writer = options["memory_enabled"] or options["local_heat_weight"] > 0
        self.writer = RGBEvidenceWriter(options["memory_dim"], options["writer_depth"], options["heatmap_size"]) if needs_writer else None
        needs_dynamic_map = options["posterior_heat_weight"] > 0 or (options["fc_weight"] > 0 and options["fc_heat_weight"] > 0)
        s_feature = config.dyn_deter // 2 + config.dyn_stoch // 2 * config.dyn_discrete
        # Consume the same initialization draws in no_semantic as in full so
        # downstream actor/critic weights start identically for the same seed.
        # The discarded head is never registered, optimized or run; no teacher
        # weights are loaded by this small random prediction-head constructor.
        dynamic_map = nn.Sequential(nn.Linear(s_feature, config.units), nn.SiLU(), nn.Linear(config.units, options["heatmap_size"] ** 2))
        self.task_heatmap = dynamic_map if needs_dynamic_map else None
        del dynamic_map
        self._scales = {"task_reward": config.reward_head["loss_scale"] * options["task_reward_weight"],
                        "clip_reward": config.reward_head["loss_scale"] * options["clip_reward_weight"],
                        "cont": config.cont_head["loss_scale"]}
        # Construct after every trainable child is registered; optimizer and
        # checkpoints therefore include writer, attention, joint dynamics/heads.
        self._model_opt = tools.Optimizer("model", self.parameters(), config.model_lr,
                                         config.opt_eps, config.grad_clip,
                                         config.weight_decay, opt=config.opt,
                                         use_amp=self._use_amp)

    def preprocess(self, obs):
        data = {key: torch.as_tensor(value, device=self._config.device, dtype=torch.float32)
                for key, value in obs.items()}
        data["image"] = data["image"] / 255.0
        if "is_first" not in data or "is_terminal" not in data:
            raise ValueError("Evidence observations require actual episode reset/terminal flags")
        data["cont"] = (1.0 - data["is_terminal"]).unsqueeze(-1)
        if "discount" in data:
            data["discount"] = (data["discount"] * self._config.discount).unsqueeze(-1)
        for name in ("reward", "task_reward", "clip_reward"):
            if name in data and data[name].ndim == data["is_first"].ndim:
                data[name] = data[name].unsqueeze(-1)
        return data

    def observe_step(self, prev_state, prev_action, embed, is_first, normalized_image, sample=True):
        tokens = self.writer(normalized_image)[0] if self.options["memory_enabled"] else None
        return self.dynamics.obs_step(prev_state, prev_action, embed, is_first, sample=sample, memory_tokens=tokens)

    def heatmap_logits(self, state):
        if self.task_heatmap is None:
            raise RuntimeError("Task heatmap head is disabled in this named ablation")
        feature = self.dynamics.s_feature(state)
        size = self.options["heatmap_size"]
        return self.task_heatmap(feature).reshape(*feature.shape[:-1], size, size)

    def policy_reward(self, feat):
        # Use exactly the same decoded task prediction in RC and behavior.
        reward = self.heads["task_reward"](feat).mode()
        if self.options["clip_scale"]:
            reward = reward + self.options["clip_scale"] * self.heads["clip_reward"](feat).mode()
        return reward

    @staticmethod
    def _masked_mean(value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        denominator = mask.expand_as(value).sum().clamp_min(1)
        return (value * mask).sum() / denominator

    def _teacher(self, data):
        if "heatmap" not in data or "heatmap_valid" not in data:
            raise ValueError("Enabled semantic losses require explicit teacher labels and heatmap_valid")
        target = data["heatmap"].detach() / 255.0
        if target.ndim == 5 and target.shape[-1] == 1:
            target = target[..., 0]
        if target.ndim != 4:
            raise ValueError("Teacher map must be (batch,time,height,width) uint8 units")
        batch, length = target.shape[:2]
        size = self.options["heatmap_size"]
        target = F.interpolate(target.reshape(-1, 1, *target.shape[-2:]), (size, size), mode="area")
        return target.reshape(batch, length, size, size)

    @staticmethod
    def _valid_prefixes(data, starts, horizons):
        """Terminal endpoint is valid; any earlier end or future reset is not."""
        batch_idx, times = starts[:, 0], starts[:, 1]
        valid = data.get("sequence_valid", torch.ones_like(data["is_first"])).bool()
        end = data["is_terminal"].bool() | data.get("is_last", torch.zeros_like(data["is_first"])).bool()
        first = data["is_first"].bool()
        length = valid.shape[1]
        masks = []
        for horizon in horizons:
            good = (times + horizon < length) & valid[batch_idx, times] & ~end[batch_idx, times]
            for offset in range(1, horizon + 1):
                index = (times + offset).clamp(max=length - 1)
                good = good & valid[batch_idx, index] & ~first[batch_idx, index]
                if offset < horizon:
                    good = good & ~end[batch_idx, index]
            masks.append(good)
        return torch.stack(masks, -1)

    def _select_starts(self, data):
        valid = data["loss_mask"].bool() & data["sequence_valid"].bool()
        valid = valid & ~data["is_terminal"].bool() & ~data.get("is_last", torch.zeros_like(data["is_first"])).bool()
        starts = []
        for batch in range(valid.shape[0]):
            times = valid[batch].nonzero(as_tuple=False).flatten()
            if not times.numel():
                continue
            candidates = torch.stack([torch.full_like(times, batch), times], -1)
            prefix = self._valid_prefixes(data, candidates, self.options["horizons"])
            candidates = candidates[prefix.any(-1)]
            if candidates.shape[0] > self.options["starts_per_sequence"]:
                permutation = torch.randperm(candidates.shape[0], device=candidates.device)
                candidates = candidates[permutation[:self.options["starts_per_sequence"]]]
            if candidates.numel():
                starts.append(candidates)
        return torch.cat(starts, 0) if starts else torch.empty(0, 2, dtype=torch.long, device=valid.device)

    def _auxiliary_losses(self, post, data):
        """Sampled-path FC surrogate and MC-averaged pure-task prefix RC."""
        options = self.options
        zero = post["deter"].new_zeros(())
        enabled_fc, enabled_rc = options["fc_weight"] > 0, options["rc_weight"] > 0
        losses = {"fc": zero, "rc": zero}
        metrics = {"fc_prefixes": zero, "rc_prefixes": zero,
                   "fc_categorical_loss": zero, "fc_hidden_loss": zero,
                   "future_heatmap_loss": zero, "rc_positive_prefixes": zero,
                   "rc_zero_prefixes": zero}
        # Raw sufficient statistics keep rare-positive MAE correctly weighted
        # across updates. The logger derives ratios; no examples means null.
        for name in ("rc_abs_error", "rc_positive_abs_error", "rc_zero_abs_error"):
            metrics[name + "_numerator"] = zero
            metrics[name + "_denominator"] = zero
        for horizon in options["horizons"]:
            metrics[f"rc_mae_h{horizon}_numerator"] = zero
            metrics[f"rc_mae_h{horizon}_denominator"] = zero
        # Disabled objectives do not select starts, sample trajectories, or read heads.
        if not enabled_fc and not enabled_rc:
            return losses, metrics
        starts = self._select_starts(data)
        if not starts.numel():
            return losses, metrics
        horizons = list(options["horizons"])
        prefix = self._valid_prefixes(data, starts, horizons)
        batch_idx, times = starts[:, 0], starts[:, 1]
        active_horizons = [k for j, k in enumerate(horizons) if prefix[:, j].any() and options["horizon_weights"][j] > 0]
        if not active_horizons:
            return losses, metrics
        maximum = max(active_horizons)
        origin = {key: value[batch_idx, times].detach() for key, value in post.items()}
        teacher = self._teacher(data) if enabled_fc and options["fc_heat_weight"] > 0 else None
        teacher_valid = data.get("heatmap_valid", torch.zeros_like(data["is_first"])).bool()
        # Keep the same estimator when RC is ablated; otherwise no_rc would
        # silently change FC's sampling budget in addition to its own weight.
        samples = options["mc_samples"]
        fc_sum, fc_count = zero, zero
        category_sum, hidden_sum, map_sum, map_count = zero, zero, zero, zero
        predicted_returns = {k: [] for k in active_horizons}
        for _ in range(samples):
            state = origin
            cumulative = zero.expand(starts.shape[0])
            for offset in range(1, maximum + 1):
                endpoint = (times + offset).clamp(max=data["is_first"].shape[1] - 1)
                state = self.dynamics.img_step(state, data["action"][batch_idx, endpoint])
                if enabled_rc:
                    frozen = _freeze_parameters(self.heads["task_reward"]) if options["freeze_rc_reward_head"] else contextlib.nullcontext()
                    with frozen:
                        predicted = self.heads["task_reward"](self.dynamics.get_feat(state)).mode()[..., 0]
                    cumulative = cumulative + self._config.discount ** (offset - 1) * predicted
                if offset not in active_horizons:
                    continue
                horizon_index = horizons.index(offset)
                mask = prefix[:, horizon_index]
                weight = float(options["horizon_weights"][horizon_index])
                if enabled_rc:
                    predicted_returns[offset].append(cumulative)
                if enabled_fc:
                    target_logit = post["logit"][batch_idx, endpoint, :self.dynamics._stoch // 2].detach()
                    prediction_logit = state["logit"][:, :self.dynamics._stoch // 2]
                    categorical = torch.distributions.kl_divergence(self.dynamics.s.get_dist({"logit": target_logit}), self.dynamics.s.get_dist({"logit": prediction_logit})) / (self.dynamics._stoch // 2)
                    category_sum = category_sum + weight * (categorical.detach() * mask).sum()
                    per_prefix = categorical
                    if options["fc_hidden_weight"] > 0:
                        target_hidden = post["deter"][batch_idx, endpoint, :self.dynamics._deter // 2].detach()
                        hidden = state["deter"][:, :self.dynamics._deter // 2]
                        hidden_error = (F.normalize(hidden, dim=-1) - F.normalize(target_hidden, dim=-1)).square().mean(-1)
                        hidden_sum = hidden_sum + weight * (hidden_error.detach() * mask).sum()
                        per_prefix = per_prefix + options["fc_hidden_weight"] * hidden_error
                    fc_sum = fc_sum + weight * (per_prefix * mask).sum()
                    if teacher is not None:
                        map_mask = mask & teacher_valid[batch_idx, endpoint]
                        map_error = (self.heatmap_logits(state).sigmoid() - teacher[batch_idx, endpoint]).square().mean((-1, -2))
                        map_sum = map_sum + weight * (map_error.detach() * map_mask).sum()
                        map_count = map_count + weight * map_mask.sum()
                        fc_sum = fc_sum + weight * options["fc_heat_weight"] * (map_error * map_mask).sum()
                    fc_count = fc_count + weight * mask.sum()
        if enabled_fc:
            losses["fc"] = fc_sum / fc_count.clamp_min(1)
            metrics["fc_prefixes"] = prefix.sum().detach().float()
            metrics["fc_categorical_loss"] = category_sum / fc_count.clamp_min(1)
            metrics["fc_hidden_loss"] = hidden_sum / fc_count.clamp_min(1)
            metrics["future_heatmap_loss"] = map_sum / map_count.clamp_min(1)
        if enabled_rc:
            rc_sum, rc_count, abs_sum = zero, zero, zero
            positive_sum, zero_sum, positive_count, zero_count = zero, zero, zero, zero
            positive_examples, zero_examples = zero, zero
            for horizon in active_horizons:
                index = horizons.index(horizon)
                mask = prefix[:, index]
                weight = float(options["horizon_weights"][index])
                actual = zero.expand(starts.shape[0])
                for offset in range(1, horizon + 1):
                    endpoint = (times + offset).clamp(max=data["is_first"].shape[1] - 1)
                    actual = actual + self._config.discount ** (offset - 1) * data["task_reward"][batch_idx, endpoint, 0]
                predicted = torch.stack(predicted_returns[horizon], 0).mean(0)
                error = predicted - actual.detach()
                positive_mask = mask & (actual.detach().abs() > 1e-6)
                zero_mask = mask & ~positive_mask
                positive_sum = positive_sum + weight * (error.detach().abs() * positive_mask).sum()
                zero_sum = zero_sum + weight * (error.detach().abs() * zero_mask).sum()
                positive_count = positive_count + weight * positive_mask.sum()
                zero_count = zero_count + weight * zero_mask.sum()
                positive_examples = positive_examples + positive_mask.sum()
                zero_examples = zero_examples + zero_mask.sum()
                metrics[f"rc_mae_h{horizon}_numerator"] = (error.detach().abs() * mask).sum()
                metrics[f"rc_mae_h{horizon}_denominator"] = mask.sum().detach().float()
                metrics[f"rc_prefixes_h{horizon}"] = mask.sum().detach().float()
                rc_sum = rc_sum + weight * (error.square() * mask).sum()
                abs_sum = abs_sum + weight * (error.abs() * mask).sum()
                rc_count = rc_count + weight * mask.sum()
            losses["rc"] = rc_sum / rc_count.clamp_min(1)
            metrics["rc_prefixes"] = prefix.sum().detach().float()
            metrics["rc_abs_error_numerator"] = abs_sum.detach()
            metrics["rc_abs_error_denominator"] = rc_count.detach().float()
            metrics["rc_positive_prefixes"] = positive_examples.detach().float()
            metrics["rc_zero_prefixes"] = zero_examples.detach().float()
            metrics["rc_positive_abs_error_numerator"] = positive_sum.detach()
            metrics["rc_positive_abs_error_denominator"] = positive_count.detach().float()
            metrics["rc_zero_abs_error_numerator"] = zero_sum.detach()
            metrics["rc_zero_abs_error_denominator"] = zero_count.detach().float()
        return losses, metrics

    def _train(self, data):
        data = self.preprocess(data)
        if "task_reward" not in data:
            raise ValueError("RC and task policy reward require separately recorded task_reward")
        if "clip_reward" in self.heads and "clip_reward" not in data:
            raise ValueError("The clip policy head requires separately recorded clip_reward")
        shape = data["is_first"].shape
        data.setdefault("sequence_valid", torch.ones(shape, device=data["image"].device))
        data.setdefault("loss_mask", data["sequence_valid"])
        mask = data["loss_mask"] * data["sequence_valid"]
        if not mask.any():
            raise ValueError("Evidence batch has no valid learning rows after burn-in")
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                tokens, local_logits = self.writer(data["image"]) if self.writer is not None else (None, None)
                post, prior = self.dynamics.observe(embed, data["action"], data["is_first"], memory_tokens=tokens if self.options["memory_enabled"] else None,
                                                    sequence_valid=data["sequence_valid"], loss_mask=mask)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(post, prior, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
                feature = self.dynamics.get_feat(post)
                losses = {}
                for name, head in self.heads.items():
                    grad_name = "reward" if name in ("task_reward", "clip_reward") else name
                    inp = feature if grad_name in self._config.grad_heads else feature.detach()
                    prediction = head(inp)
                    predictions = prediction if isinstance(prediction, dict) else {name: prediction}
                    for key, distribution in predictions.items():
                        losses[key] = self._masked_mean(-distribution.log_prob(data[key]), mask)
                model_loss = self._masked_mean(kl_loss, mask) + sum(loss * self._scales.get(key, 1.0) for key, loss in losses.items())
                semantic = self.options["local_heat_weight"] > 0 or self.options["posterior_heat_weight"] > 0
                if semantic:
                    target = self._teacher(data)
                    map_mask = mask * data["heatmap_valid"]
                    if self.options["local_heat_weight"] > 0:
                        losses["local_heat"] = self._masked_mean((local_logits.sigmoid() - target).square().mean((-1, -2)), map_mask)
                        model_loss = model_loss + self.options["local_heat_weight"] * losses["local_heat"]
                    if self.options["posterior_heat_weight"] > 0:
                        losses["posterior_heat"] = self._masked_mean((self.heatmap_logits(post).sigmoid() - target).square().mean((-1, -2)), map_mask)
                        model_loss = model_loss + self.options["posterior_heat_weight"] * losses["posterior_heat"]
                auxiliary, auxiliary_metrics = self._auxiliary_losses(post, data)
                model_loss = model_loss + self.options["fc_weight"] * auxiliary["fc"] + self.options["rc_weight"] * auxiliary["rc"]
            metrics = self._model_opt(model_loss, self.parameters())
        def scalar(value):
            return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)
        metrics.update({f"{name}_loss": scalar(loss) for name, loss in losses.items()})
        metrics.update({"fc_loss": scalar(auxiliary["fc"]), "rc_loss": scalar(auxiliary["rc"]),
                        "kl": scalar(self._masked_mean(kl_value, mask)), "dyn_loss": scalar(self._masked_mean(dyn_loss, mask)),
                        "rep_loss": scalar(self._masked_mean(rep_loss, mask)),
                        "prior_ent": scalar(self._masked_mean(self.dynamics.get_dist(prior).entropy(), mask)),
                        "post_ent": scalar(self._masked_mean(self.dynamics.get_dist(post).entropy(), mask)),
                        "memory_fill": scalar(self._masked_mean(post["mem_valid"].float().mean(-1), mask)),
                        "learning_rows": scalar(mask.sum()), "semantic_rows": scalar((mask * data.get("heatmap_valid", torch.zeros_like(mask))).sum()),
                        "kl_free": self._config.kl_free, "dyn_scale": self._config.dyn_scale, "rep_scale": self._config.rep_scale})
        metrics.update({name: scalar(value) for name, value in auxiliary_metrics.items()})
        context = {"embed": embed.detach(), "feat": feature.detach(), "kl": kl_value.detach(),
                   "postent": self.dynamics.get_dist(post).entropy().detach(),
                   "loss_mask": mask.detach(), "sequence_valid": data["sequence_valid"].detach()}
        return {key: value.detach() for key, value in post.items()}, context, metrics

    def diagnostic_metrics(self, batch, gradients=True, starts_per_sequence=None):
        """Measure a fixed recorded batch without optimizing any parameter.

        The caller owns a persisted probe batch and isolates RNG/module modes.
        Disabled FC/RC are still measured here; their configured gradient
        contributions remain zero. This does not change the training path.
        """
        from .diagnostic_model import diagnostic_metrics
        return diagnostic_metrics(self, batch, gradients=gradients,
                                  starts_per_sequence=starts_per_sequence)

