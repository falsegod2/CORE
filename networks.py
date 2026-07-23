import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
    """Standard single-stream DreamerV3 recurrent state-space model.

    State dictionary:
      - deter: deterministic recurrent state h_t
      - stoch: stochastic categorical/continuous state z_t
      - logit, or mean/std: distribution parameters
    """

    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super().__init__()
        if num_actions is None or embed is None or device is None:
            raise ValueError("num_actions, embed, and device must be provided")

        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        act_fn = getattr(torch.nn, act)

        stoch_size = self._stoch * self._discrete if self._discrete else self._stoch
        self._img_in_layers = nn.Sequential(
            nn.Linear(stoch_size + self._num_actions, self._hidden, bias=False),
            *([nn.LayerNorm(self._hidden, eps=1e-3)] if norm else []),
            act_fn(),
        )
        self._img_in_layers.apply(tools.weight_init)

        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        self._img_out_layers = nn.Sequential(
            nn.Linear(self._deter, self._hidden, bias=False),
            *([nn.LayerNorm(self._hidden, eps=1e-3)] if norm else []),
            act_fn(),
        )
        self._img_out_layers.apply(tools.weight_init)

        self._obs_out_layers = nn.Sequential(
            nn.Linear(self._deter + self._embed, self._hidden, bias=False),
            *([nn.LayerNorm(self._hidden, eps=1e-3)] if norm else []),
            act_fn(),
        )
        self._obs_out_layers.apply(tools.weight_init)

        stat_size = self._stoch * self._discrete if self._discrete else 2 * self._stoch
        self._imgs_stat_layer = nn.Linear(self._hidden, stat_size)
        self._obs_stat_layer = nn.Linear(self._hidden, stat_size)
        self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
        self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device))
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = {
                "logit": torch.zeros(
                    batch_size, self._stoch, self._discrete, device=self._device
                ),
                "stoch": torch.zeros(
                    batch_size, self._stoch, self._discrete, device=self._device
                ),
                "deter": deter,
            }
        else:
            state = {
                "mean": torch.zeros(batch_size, self._stoch, device=self._device),
                "std": torch.zeros(batch_size, self._stoch, device=self._device),
                "stoch": torch.zeros(batch_size, self._stoch, device=self._device),
                "deter": deter,
            }

        if self._initial == "zeros":
            return state
        if self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            stats = self._suff_stats_layer("ims", self._img_out_layers(state["deter"]))
            state.update(stats)
            return state
        raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev, prev_action, current_embed, first: self.obs_step(
                prev[0], prev_action, current_embed, first
            ),
            (action, embed, is_first),
            (state, state),
        )
        return (
            {key: swap(value) for key, value in post.items()},
            {key: swap(value) for key, value in prior.items()},
        )

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)[0]
        return {key: swap(value) for key, value in prior.items()}

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            stoch = stoch.reshape(
                list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            )
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        del dtype
        if self._discrete:
            return torchd.independent.Independent(
                tools.OneHotDist(
                    state["logit"], unimix_ratio=self._unimix_ratio
                ),
                1,
            )
        return tools.ContDist(
            torchd.independent.Independent(
                torchd.normal.Normal(state["mean"], state["std"]), 1
            )
        )

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        batch_size = len(is_first)
        if prev_state is None or torch.sum(is_first) == batch_size:
            prev_state = self.initial(batch_size)
            prev_action = torch.zeros(
                batch_size, self._num_actions, device=self._device
            )
        elif torch.sum(is_first) > 0:
            first = is_first[:, None]
            prev_action = prev_action * (1.0 - first)
            initial = self.initial(batch_size)
            prev_state = dict(prev_state)
            for key, value in prev_state.items():
                first_r = torch.reshape(
                    first,
                    first.shape + (1,) * (len(value.shape) - len(first.shape)),
                )
                prev_state[key] = value * (1.0 - first_r) + initial[key] * first_r

        prior = self.img_step(prev_state, prev_action, sample=sample)
        x = self._obs_out_layers(torch.cat([prior["deter"], embed], -1))
        stats = self._suff_stats_layer("obs", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            prev_stoch = prev_stoch.reshape(
                list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            )
        x = self._img_in_layers(torch.cat([prev_stoch, prev_action], -1))
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter_state = self._cell(x, [deter])
            deter = deter_state[0]
        x = self._img_out_layers(x)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        return {"stoch": stoch, "deter": deter, **stats}

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        return self.get_dist(stats).mode()

    def _suff_stats_layer(self, name, x):
        if name == "ims":
            x = self._imgs_stat_layer(x)
        elif name == "obs":
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError(name)

        if self._discrete:
            return {
                "logit": x.reshape(
                    list(x.shape[:-1]) + [self._stoch, self._discrete]
                )
            }

        mean, std = torch.split(x, [self._stoch] * 2, -1)
        mean = {
            "none": lambda: mean,
            "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
        }[self._mean_act]()
        std = {
            "softplus": lambda: F.softplus(std),
            "abs": lambda: torch.abs(std + 1.0),
            "sigmoid": lambda: torch.sigmoid(std),
            "sigmoid2": lambda: 2.0 * torch.sigmoid(std / 2.0),
        }[self._std_act]()
        return {"mean": mean, "std": std + self._min_std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        stop_gradient = lambda state: {
            key: value.detach() for key, value in state.items()
        }

        post_dist = self.get_dist(post)
        prior_sg_dist = self.get_dist(stop_gradient(prior))
        post_sg_dist = self.get_dist(stop_gradient(post))
        prior_dist = self.get_dist(prior)

        if self._discrete:
            rep_loss = value = kld(post_dist, prior_sg_dist)
            dyn_loss = kld(post_sg_dist, prior_dist)
        else:
            rep_loss = value = kld(post_dist._dist, prior_sg_dist._dist)
            dyn_loss = kld(post_sg_dist._dist, prior_dist._dist)

        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class TaskRelevantObjectEncoder(nn.Module):
    """Task-guided object tokens built from Dreamer RGB patch features.

    This is deliberately not pure unsupervised Slot Attention. The frozen
    MineCLIP task embedding first identifies task-relevant spatial candidates.
    A small set of task-conditioned object queries then aggregates only those
    candidates. The resulting object tokens are projected back to the spatial
    RGB token grid through a conservative residual path, so the RSSM input size
    and the baseline Dreamer visual stream remain unchanged.
    """

    def __init__(self, shapes, encoder_config, object_config):
        super().__init__()
        self._base = MultiEncoder(shapes, **encoder_config)
        if not self._base.cnn_shapes or not hasattr(self._base, "_cnn"):
            raise ValueError("Task object tokens require an RGB CNN encoder")
        if "image" not in self._base.cnn_shapes:
            raise ValueError("Task object tokens expect the 'image' observation")

        self.outdim = self._base.outdim
        self._task_key = object_config.get("task_key", "task_embedding")
        self._task_dim = int(object_config.get("task_dim", 512))
        self._visual_key = object_config.get(
            "visual_key", "mineclip_embedding"
        )
        self._visual_dim = int(object_config.get("visual_dim", 512))
        self._object_dim = int(object_config.get("object_dim", 256))
        self._hidden = int(object_config.get("hidden", 256))
        self._num_objects = int(object_config.get("num_objects", 4))
        self._candidate_topk = int(object_config.get("candidate_topk", 8))
        self._object_iters = int(object_config.get("object_iters", 2))
        self._relevance_temperature = float(
            object_config.get("relevance_temperature", 0.2)
        )
        self._attention_temperature = float(
            object_config.get("attention_temperature", 0.5)
        )
        self._relevance_bias = float(
            object_config.get("relevance_bias", 2.0)
        )
        self._residual_scale = float(
            object_config.get("residual_scale", 0.05)
        )
        self._normalize_task = bool(
            object_config.get("normalize_task", True)
        )
        self._coverage_scale = float(
            object_config.get("coverage_scale", 0.02)
        )
        self._diversity_scale = float(
            object_config.get("diversity_scale", 0.005)
        )
        self._semantic_align_scale = float(
            object_config.get("semantic_align_scale", 0.02)
        )
        output_init = float(object_config.get("output_init", 1e-3))

        shape = tuple(shapes.get(self._task_key, ()))
        if shape != (self._task_dim,):
            raise ValueError(
                f"Observation '{self._task_key}' must have shape "
                f"({self._task_dim},), got {shape}"
            )
        visual_shape = tuple(shapes.get(self._visual_key, ()))
        if visual_shape != (self._visual_dim,):
            raise ValueError(
                f"Observation '{self._visual_key}' must have shape "
                f"({self._visual_dim},), got {visual_shape}"
            )
        if self._num_objects < 1:
            raise ValueError("num_objects must be at least 1")
        if self._object_iters < 1:
            raise ValueError("object_iters must be at least 1")
        if self._relevance_temperature <= 0:
            raise ValueError("relevance_temperature must be positive")
        if self._attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")

        token_dim = self._base._cnn.token_dim
        token_count = self._base._cnn.token_count
        if not 1 <= self._candidate_topk <= token_count:
            raise ValueError(
                f"candidate_topk must be in [1, {token_count}], "
                f"got {self._candidate_topk}"
            )
        self._token_count = token_count
        self._token_dim = token_dim

        self._patch_key = nn.Linear(token_dim, self._object_dim, bias=False)
        self._patch_value = nn.Linear(token_dim, self._object_dim, bias=False)
        self._task_proj = nn.Linear(
            self._task_dim, self._object_dim, bias=False
        )
        self._visual_proj = nn.Linear(
            self._visual_dim, self._object_dim, bias=False
        )
        self._task_to_objects = nn.Linear(
            self._task_dim,
            self._num_objects * self._object_dim,
            bias=False,
        )
        self._visual_to_objects = nn.Linear(
            self._visual_dim,
            self._num_objects * self._object_dim,
            bias=False,
        )
        self._object_query = nn.Linear(
            self._object_dim, self._object_dim, bias=False
        )
        self._object_update = nn.Sequential(
            nn.Linear(2 * self._object_dim, self._hidden),
            nn.LayerNorm(self._hidden, eps=1e-3),
            nn.SiLU(),
            nn.Linear(self._hidden, self._object_dim),
        )
        self._object_norm = nn.LayerNorm(self._object_dim, eps=1e-3)
        self._patch_norm = nn.LayerNorm(self._object_dim, eps=1e-3)
        self._task_norm = nn.LayerNorm(self._object_dim, eps=1e-3)
        self._object_to_rgb = nn.Linear(
            self._object_dim, token_dim, bias=False
        )
        self._global_to_rgb = nn.Linear(
            self._object_dim, token_dim, bias=False
        )
        self._object_semantic = nn.Linear(
            self._object_dim, self._visual_dim, bias=False
        )
        self._patch_gate = nn.Sequential(
            nn.Linear(2 * self._object_dim, self._hidden),
            nn.LayerNorm(self._hidden, eps=1e-3),
            nn.SiLU(),
            nn.Linear(self._hidden, token_dim),
        )

        self._object_seed = nn.Parameter(
            torch.zeros(1, self._num_objects, self._object_dim)
        )
        self._position = nn.Parameter(
            torch.zeros(1, token_count, self._object_dim)
        )

        for module in (
            self._patch_key,
            self._patch_value,
            self._task_proj,
            self._visual_proj,
            self._task_to_objects,
            self._visual_to_objects,
            self._object_query,
            self._object_update,
            self._patch_gate,
        ):
            module.apply(tools.weight_init)
        nn.init.trunc_normal_(self._object_seed, std=0.02)
        nn.init.trunc_normal_(self._position, std=0.02)
        nn.init.normal_(self._object_to_rgb.weight, mean=0.0, std=output_init)
        nn.init.normal_(self._global_to_rgb.weight, mean=0.0, std=output_init)
        self._object_semantic.apply(tools.weight_init)
        nn.init.constant_(self._patch_gate[-1].bias, -2.0)

        self._last_metrics = {}
        self._last_aux_losses = {}

    def _broadcast_parameter(self, parameter, prefix_rank):
        return parameter.reshape((1,) * prefix_rank + parameter.shape[1:])

    def forward(self, obs):
        cnn_inputs = torch.cat(
            [obs[key] for key in self._base.cnn_shapes], dim=-1
        )
        raw_tokens = self._base._cnn.forward_tokens(cnn_inputs)
        prefix = raw_tokens.shape[:-2]
        prefix_rank = len(prefix)
        token_count, token_dim = raw_tokens.shape[-2:]

        task = obs[self._task_key].float().detach()
        visual_semantic = obs[self._visual_key].float().detach()
        visual_valid = (
            torch.linalg.vector_norm(visual_semantic, dim=-1) > 1e-6
        ).to(raw_tokens.dtype)
        if self._normalize_task:
            task = F.normalize(task, dim=-1, eps=1e-6)
            visual_semantic = F.normalize(
                visual_semantic, dim=-1, eps=1e-6
            )

        patch_key = self._patch_norm(self._patch_key(raw_tokens))
        patch_key = patch_key + self._broadcast_parameter(
            self._position, prefix_rank
        )
        patch_value = self._patch_value(raw_tokens)
        task_query = self._task_norm(
            self._task_proj(task) + self._visual_proj(visual_semantic)
        )

        patch_unit = F.normalize(patch_key, dim=-1, eps=1e-6)
        task_unit = F.normalize(task_query, dim=-1, eps=1e-6)
        relevance_logits = torch.sum(
            patch_unit * task_unit.unsqueeze(-2), dim=-1
        ) / self._relevance_temperature
        relevance = torch.softmax(relevance_logits, dim=-1)

        top_values, top_indices = torch.topk(
            relevance, self._candidate_topk, dim=-1
        )
        candidate_mask = torch.zeros_like(relevance).scatter(
            -1, top_indices, 1.0
        )
        candidate_weights = relevance * candidate_mask
        candidate_mass = candidate_weights.sum(dim=-1, keepdim=True)
        candidate_weights = candidate_weights / (candidate_mass + 1e-8)

        task_objects = (
            self._task_to_objects(task)
            + self._visual_to_objects(visual_semantic)
        ).reshape(prefix + (self._num_objects, self._object_dim))
        object_seed = self._broadcast_parameter(
            self._object_seed, prefix_rank
        )
        objects = self._object_norm(object_seed + task_objects)

        mask_value = torch.finfo(raw_tokens.dtype).min
        attention = None
        for _ in range(self._object_iters):
            object_query = F.normalize(
                self._object_query(objects), dim=-1, eps=1e-6
            )
            key_unit = F.normalize(patch_key, dim=-1, eps=1e-6)
            # Cosine attention: queries and keys are unit-normalized, so the
            # explicit temperature (rather than an additional sqrt(d) factor)
            # controls how strongly different object queries specialize.
            logits = torch.einsum(
                "...kd,...nd->...kn", object_query, key_unit
            ) / self._attention_temperature
            relevance_bias = self._relevance_bias * torch.log(
                candidate_weights + 1e-8
            )
            logits = logits + relevance_bias.unsqueeze(-2)
            logits = torch.where(
                candidate_mask.unsqueeze(-2).bool(),
                logits,
                torch.full_like(logits, mask_value),
            )
            attention = torch.softmax(logits, dim=-1)
            context = torch.einsum(
                "...kn,...nd->...kd", attention, patch_value
            )
            update = self._object_update(
                torch.cat([objects, context], dim=-1)
            )
            objects = self._object_norm(objects + update)

        # A scene token preserves task-relevant context that is shared across
        # objects, while the object tokens specialize through distinct queries.
        global_token = torch.einsum(
            "...n,...nd->...d", candidate_weights, patch_value
        )

        object_rgb = self._object_to_rgb(objects)
        patch_object_weights = attention.transpose(-1, -2)
        patch_object_weights = patch_object_weights / (
            patch_object_weights.sum(dim=-1, keepdim=True) + 1e-8
        )
        object_delta = torch.einsum(
            "...nk,...kc->...nc", patch_object_weights, object_rgb
        )
        global_delta = self._global_to_rgb(global_token).unsqueeze(-2)
        global_delta = candidate_weights.unsqueeze(-1) * global_delta

        task_tokens = task_query.unsqueeze(-2).expand_as(patch_key)
        gate = torch.sigmoid(
            self._patch_gate(torch.cat([patch_key, task_tokens], dim=-1))
        )
        applied_delta = self._residual_scale * gate * (
            object_delta + global_delta
        )
        refined_tokens = raw_tokens + applied_delta
        cnn_embed = refined_tokens.reshape(
            prefix + (token_count * token_dim,)
        )

        outputs = [cnn_embed]
        if self._base.mlp_shapes:
            mlp_inputs = torch.cat(
                [obs[key] for key in self._base.mlp_shapes], dim=-1
            )
            outputs.append(self._base._mlp(mlp_inputs))
        fused = torch.cat(outputs, dim=-1)

        # Auxiliary terms are weak and local to object binding. Coverage keeps
        # the object set on task-relevant candidates; diversity prevents every
        # object query from collapsing onto the same patch.
        coverage = attention.sum(dim=-2)
        coverage = coverage / (coverage.sum(dim=-1, keepdim=True) + 1e-8)
        coverage_loss = -torch.sum(
            candidate_weights * torch.log(coverage + 1e-8), dim=-1
        )
        if self._num_objects > 1:
            attn_unit = F.normalize(attention, p=2, dim=-1, eps=1e-6)
            gram = torch.einsum(
                "...kn,...jn->...kj", attn_unit, attn_unit
            )
            eye = torch.eye(
                self._num_objects,
                device=gram.device,
                dtype=gram.dtype,
            ).reshape(
                (1,) * len(prefix)
                + (self._num_objects, self._num_objects)
            )
            off_diagonal = gram * (1.0 - eye)
            diversity_loss = torch.sum(
                off_diagonal ** 2, dim=(-2, -1)
            ) / (self._num_objects * (self._num_objects - 1))
            overlap = torch.sum(
                off_diagonal.detach(), dim=(-2, -1)
            ) / (self._num_objects * (self._num_objects - 1))
        else:
            diversity_loss = torch.zeros_like(coverage_loss)
            overlap = torch.zeros_like(coverage_loss)

        object_summary = objects.mean(dim=-2)
        predicted_semantic = F.normalize(
            self._object_semantic(object_summary), dim=-1, eps=1e-6
        )
        target_semantic = F.normalize(
            visual_semantic, dim=-1, eps=1e-6
        )
        semantic_cosine = torch.sum(
            predicted_semantic * target_semantic, dim=-1
        )
        semantic_align_loss = (1.0 - semantic_cosine) * visual_valid

        self._last_aux_losses = {
            "task_object_coverage": self._coverage_scale * coverage_loss,
            "task_object_diversity": self._diversity_scale * diversity_loss,
            "task_object_semantic_align": (
                self._semantic_align_scale * semantic_align_loss
            ),
        }

        with torch.no_grad():
            raw_rms = torch.sqrt(torch.mean(raw_tokens.detach() ** 2) + 1e-8)
            delta_rms = torch.sqrt(
                torch.mean(applied_delta.detach() ** 2) + 1e-8
            )
            relevance_entropy = -torch.sum(
                relevance.detach() * torch.log(relevance.detach() + 1e-8),
                dim=-1,
            )
            object_entropy = -torch.sum(
                attention.detach() * torch.log(attention.detach() + 1e-8),
                dim=-1,
            )
            self._last_metrics = {
                "task_object_relevance_entropy": relevance_entropy.mean(),
                "task_object_top1": top_values[..., 0].detach().mean(),
                "task_object_candidate_mass": candidate_mass.detach().mean(),
                "task_object_attention_entropy": object_entropy.mean(),
                "task_object_attention_overlap": overlap.mean(),
                "task_object_gate_mean": gate.detach().mean(),
                "task_object_delta_ratio": delta_rms / raw_rms,
                "task_object_coverage_loss": coverage_loss.detach().mean(),
                "task_object_diversity_loss": diversity_loss.detach().mean(),
                "task_object_semantic_cosine": (
                    semantic_cosine.detach() * visual_valid
                ).sum() / (visual_valid.sum() + 1e-8),
                "task_object_semantic_valid": visual_valid.detach().mean(),
            }
        return fused

    def get_metrics(self):
        return dict(self._last_metrics)

    def get_aux_losses(self):
        return dict(self._last_aux_losses)


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.token_dim = out_dim // 2
        self.token_height = h
        self.token_width = w
        self.token_count = h * w
        self.outdim = self.token_dim * self.token_count
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward_tokens(self, obs):
        """Return the final spatial CNN map as patch tokens [..., N, C]."""
        prefix_shape = list(obs.shape[:-3])
        x = obs - 0.5
        x = x.reshape((-1,) + tuple(obs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # N = H' * W'; preserve spatial token identity until task selection.
        x = x.permute(0, 2, 3, 1).reshape(
            x.shape[0], self.token_count, self.token_dim
        )
        return x.reshape(prefix_shape + [self.token_count, self.token_dim])

    def forward(self, obs):
        tokens = self.forward_tokens(obs)
        return tokens.reshape(list(obs.shape[:-3]) + [self.outdim])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            if self._dist == "normal_std_fixed":
                mean = torch.sigmoid(mean)
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
