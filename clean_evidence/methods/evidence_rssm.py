"""Joint action-conditioned RSSM with causal, observation-only evidence.

Memory tensors are part of the ordinary latent-state dictionary. Consequently
the pinned Dreamer imagination code snapshots and carries exactly the same
evidence as the recorded-action prediction branch. ``img_step`` never writes
evidence; only ``obs_step`` may append tokens from a newly received RGB image.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..vendor.dreamerv3_torch import tools
from .dual_rssm_torch import DualRSSM


class _MemoryDiagnostics:
    """Detached sufficient statistics from an explicitly requested probe.

    Read measurements include only rows whose memory actually influences the
    transition. Thus an empty buffer or a skipped start-only read has no gate,
    attention or residual observation, rather than a misleading zero error.
    Stored values are Python numbers, never live autograd tensors.
    """
    _ratios = (
        "available_fraction", "active_read_fraction", "gate_mean",
        "gate_low_fraction", "gate_high_fraction", "attention_null_mass",
        "attention_real_mass", "attention_entropy_normalized",
        "attended_age_steps", "gated_delta_to_state_norm",
        "effective_delta_to_state_norm",
    )

    def __init__(self, prefix, enabled, read_mode):
        self.prefix = prefix
        self.stats = {
            prefix + "_enabled": int(enabled),
            prefix + "_read_mode_code": {"none": 0, "start_only": 1, "each_step": 2}[read_mode],
            prefix + "_transition_rows": 0,
            prefix + "_active_read_rows": 0,
        }
        for name in self._ratios:
            self.stats[prefix + "_" + name + "_numerator"] = 0.0
            self.stats[prefix + "_" + name + "_denominator"] = 0.0

    def add(self, name, numerator, denominator):
        key = self.prefix + "_" + name
        self.stats[key + "_numerator"] += float(numerator)
        self.stats[key + "_denominator"] += float(denominator)

    @torch.no_grad()
    def transition(self, available, active):
        available_count = int(available.sum().item())
        active_count = int(active.sum().item())
        count = available.numel()
        self.stats[self.prefix + "_transition_rows"] += count
        self.stats[self.prefix + "_active_read_rows"] += active_count
        self.add("available_fraction", available_count, count)
        self.add("active_read_fraction", active_count, count)

    @torch.no_grad()
    def read(self, model, query, keys, valid, raw_age, use, gate,
             base_state, gated_delta, retrieved):
        active = use.squeeze(-1).bool()
        if not bool(active.any()):
            return
        # Keep the real forward's need_weights=False call untouched. Rebuild
        # only Q/K weights from detached values; dropout is zero in this RSSM.
        # This auxiliary calculation never feeds the state or optimizer.
        attention = model.attention
        with torch.autocast(device_type=query.device.type, enabled=False):
            width, heads = attention.embed_dim, attention.num_heads
            weight = attention.in_proj_weight.detach().float()
            bias = attention.in_proj_bias
            bias = None if bias is None else bias.detach().float()
            q = F.linear(query.detach().float(), weight[:width],
                         None if bias is None else bias[:width])
            k = F.linear(keys.detach().float(), weight[width:2 * width],
                         None if bias is None else bias[width:2 * width])
            q = q.reshape(q.shape[0], heads, width // heads)
            k = k.reshape(k.shape[0], k.shape[1], heads, width // heads).transpose(1, 2)
            logits = torch.einsum("bhd,bhtd->bht", q, k) / math.sqrt(width // heads)
            weights = logits.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            weights = weights[active]
            null_mass = weights[..., -1]
            real_weights = weights[..., :-1]
            real_mass = real_weights.sum(-1)
            entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
            # Includes the learned null token. Uniform attention over all
            # available keys has normalized entropy one, regardless of fill.
            key_count = valid[active].sum(-1).float()
            normalized_entropy = entropy / key_count.log()[:, None]
            age = raw_age.detach().float()[active, None, :]
            self.add("attention_null_mass", null_mass.sum().item(), null_mass.numel())
            self.add("attention_real_mass", real_mass.sum().item(), real_mass.numel())
            self.add("attention_entropy_normalized", normalized_entropy.sum().item(), normalized_entropy.numel())
            self.add("attended_age_steps", (real_weights * age).sum().item(), real_mass.sum().item())
            active_gate = gate.detach().float()[active]
            self.add("gate_mean", active_gate.sum().item(), active_gate.numel())
            self.add("gate_low_fraction", (active_gate < 0.05).sum().item(), active_gate.numel())
            self.add("gate_high_fraction", (active_gate > 0.95).sum().item(), active_gate.numel())
            base = base_state.detach().float()[active]
            norm = base.norm(dim=-1).clamp_min(1e-8)
            delta_ratio = gated_delta.detach().float()[active].norm(dim=-1) / norm
            effective_ratio = (retrieved.detach().float()[active] - base).norm(dim=-1) / norm
            self.add("gated_delta_to_state_norm", delta_ratio.sum().item(), delta_ratio.numel())
            self.add("effective_delta_to_state_norm", effective_ratio.sum().item(), effective_ratio.numel())


class EvidenceRSSM(DualRSSM):
    _latent_keys = ("deter", "stoch", "logit")
    _memory_keys = ("mem_tokens", "mem_valid", "mem_age", "imag_steps")

    def __init__(self, *args, memory_observations=16, memory_dim=256,
                 attention_heads=4, memory_enabled=True, read_mode="each_step",
                 seed=None, **kwargs):
        kwargs.pop("both_action", None)
        super().__init__(*args, both_action=True, seed=seed, **kwargs)
        if isinstance(memory_observations, bool) or not isinstance(memory_observations, int) or memory_observations < 1:
            raise ValueError("memory_observations must be a positive integer")
        if isinstance(memory_dim, bool) or not isinstance(memory_dim, int) or memory_dim < 1:
            raise ValueError("memory_dim must be a positive integer")
        if isinstance(attention_heads, bool) or not isinstance(attention_heads, int) or attention_heads < 1 or memory_dim % attention_heads:
            raise ValueError("memory_dim must be divisible by attention_heads")
        if read_mode not in ("each_step", "start_only", "none"):
            raise ValueError("read_mode must be each_step, start_only, or none")
        self.memory_observations = memory_observations
        self.memory_dim = memory_dim
        self.tokens_per_observation = 4
        self.memory_capacity = memory_observations * self.tokens_per_observation
        if not isinstance(memory_enabled, bool):
            raise ValueError("memory_enabled must be a boolean")
        self.memory_enabled = memory_enabled
        self.read_mode = read_mode
        self._memory_diagnostics = None
        self._memory_diagnostic_rows = None
        branch_deter, branch_hidden = self._deter // 2, self._hidden // 2
        branch_stoch = self._stoch // 2 * self._discrete
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((int(torch.initial_seed() if seed is None else seed) + 2) % (2**63 - 1))
            self.z_context = nn.Linear(branch_deter + branch_stoch, branch_hidden, bias=False)
            self.s_context = nn.Linear(2 * branch_deter + branch_stoch, branch_hidden, bias=False)
            self.z_context.apply(tools.weight_init)
            self.s_context.apply(tools.weight_init)
            if self.memory_enabled and read_mode != "none":
                self.query = nn.Linear(2 * branch_deter + self._num_actions, memory_dim)
                self.age_encoding = nn.Sequential(nn.Linear(1, memory_dim), nn.Tanh())
                self.attention = nn.MultiheadAttention(memory_dim, attention_heads, dropout=0.0, batch_first=True)
                self.null_evidence = nn.Parameter(torch.zeros(1, 1, memory_dim))
                self.read_out = nn.Linear(memory_dim, branch_deter, bias=False)
                self.read_gate = nn.Linear(2 * branch_deter + memory_dim, branch_deter)
                nn.init.constant_(self.read_gate.bias, -2.0)
                self.read_norm = nn.LayerNorm(branch_deter, eps=1e-3)

    @contextlib.contextmanager
    def capture_memory_diagnostics(self, prefix="memory"):
        """Yield ratio sufficient statistics for transitions in this scope.

        Modes: none=0, start_only=1, each_step=2. Empty populations have a zero
        denominator and should be reported as null, not as successful reads.
        This scope does not isolate RNG or alter model mode; the caller owns
        those protections for the complete fixed-batch diagnostic pass.
        """
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("Memory diagnostic prefix must be a nonempty string")
        if self._memory_diagnostics is not None:
            raise RuntimeError("Nested memory diagnostic captures are unsupported")
        collector = _MemoryDiagnostics(prefix, self.memory_enabled, self.read_mode)
        self._memory_diagnostics = collector
        try:
            yield collector.stats
        finally:
            self._memory_diagnostics = None

    @contextlib.contextmanager
    def memory_diagnostic_mask(self, valid_rows):
        """Limit telemetry to selected batch rows without limiting the rollout.

        Recorded-prefix probes may computationally roll all rows to their
        maximum horizon. The caller masks rows beyond their valid recorded
        prefix with this scope so they do not become memory observations.
        ``valid_rows`` is a one-dimensional boolean tensor matching each
        img_step batch in this scope. Nested scopes temporarily override the
        previous mask and restore it even on failure. No mask affects the
        latent state, attention computation, random draws or gradients.
        """
        if not isinstance(valid_rows, torch.Tensor) or valid_rows.dtype != torch.bool or valid_rows.ndim != 1:
            raise ValueError("Memory diagnostic mask must be a one-dimensional boolean tensor")
        previous = self._memory_diagnostic_rows
        self._memory_diagnostic_rows = valid_rows.detach()
        try:
            yield
        finally:
            self._memory_diagnostic_rows = previous

    def _validate_state(self, state):
        if not isinstance(state, dict) or not set(self._latent_keys).issubset(state):
            raise ValueError("EvidenceRSSM needs deter, stoch and logit")
        super()._validate_state({key: state[key] for key in self._latent_keys})
        if not set(self._memory_keys).issubset(state):
            raise ValueError("EvidenceRSSM state is missing a causal memory snapshot")
        batch = state["deter"].shape[:-1]
        if state["mem_tokens"].shape != batch + (self.memory_capacity, self.memory_dim):
            raise ValueError("Evidence memory token shape does not match the configured budget")
        if state["mem_valid"].shape != batch + (self.memory_capacity,) or state["mem_age"].shape != batch + (self.memory_capacity,):
            raise ValueError("Evidence memory validity/age shape mismatch")

    def _split_state(self, state):
        self._validate_state(state)
        left, right = {}, {}
        for key in self._latent_keys:
            left[key], right[key] = torch.chunk(state[key], 2, dim=-1 if key == "deter" else -2)
        return left, right

    def initial(self, batch_size):
        # Calling DualRSSM.initial does not consult its state validator.
        state = super().initial(batch_size)
        state["mem_tokens"] = state["deter"].new_zeros(batch_size, self.memory_capacity, self.memory_dim)
        state["mem_valid"] = torch.zeros(batch_size, self.memory_capacity, device=state["deter"].device, dtype=torch.bool)
        state["mem_age"] = state["deter"].new_zeros(batch_size, self.memory_capacity)
        state["imag_steps"] = state["deter"].new_zeros(batch_size, 1)
        return state

    @staticmethod
    def _advance(branch, previous, action, context):
        latent = previous["stoch"].reshape(previous["deter"].shape[:-1] + (-1,))
        hidden = branch._img_in_layers(torch.cat([latent, action], -1)) + context
        _, deter = branch._cell(hidden, [previous["deter"]])
        return deter[0]

    @staticmethod
    def _prior(branch, deter, sample):
        stats = branch._suff_stats_layer("ims", branch._img_out_layers(deter))
        distribution = branch.get_dist(stats)
        return dict(deter=deter, stoch=distribution.sample() if sample else distribution.mode(), **stats)

    def img_step(self, prev_state, prev_action, sample=True):
        left, right = self._split_state(prev_state)
        self._validate_action(prev_action, prev_state["deter"].shape[:-1])
        s_old = torch.cat([left["deter"], left["stoch"].flatten(-2)], -1)
        z_deter = self._advance(self.z, right, prev_action, self.z_context(s_old))
        z_context = torch.cat([right["deter"], right["stoch"].flatten(-2), z_deter], -1)
        s_deter = self._advance(self.s, left, prev_action, self.s_context(z_context))
        should_read = self.memory_enabled and self.read_mode != "none"
        # In the start-only control, later imagination transitions do not even
        # execute attention. Real obs_step resets imag_steps to zero as before.
        if should_read and self.read_mode == "start_only":
            should_read = bool((prev_state["imag_steps"] == 0).any())
        collector = self._memory_diagnostics
        diagnostic_rows = None
        if collector is not None:
            available = prev_state["mem_valid"].any(-1)
            active = available & bool(should_read)
            if self.read_mode == "start_only":
                active = active & (prev_state["imag_steps"].squeeze(-1) == 0)
            diagnostic_rows = self._memory_diagnostic_rows
            if diagnostic_rows is None:
                collector.transition(available, active)
            else:
                if diagnostic_rows.shape != available.shape:
                    raise ValueError("Memory diagnostic mask must match the img_step batch")
                diagnostic_rows = diagnostic_rows.to(device=available.device)
                collector.transition(available[diagnostic_rows], active[diagnostic_rows])
        if should_read:
            query = self.query(torch.cat([s_deter, z_deter, prev_action], -1))
            tokens = prev_state["mem_tokens"]
            # Evidence timestamps remain fixed, but its age at the predicted
            # query time advances with the imagination counter.
            age = (prev_state["mem_age"] + prev_state["imag_steps"] + 1) / float(self.memory_observations)
            keys = tokens + self.age_encoding(age[..., None])
            null = self.null_evidence.expand(tokens.shape[0], -1, -1)
            keys = torch.cat([keys, null], -2)
            values = torch.cat([tokens, null], -2)
            valid = torch.cat([prev_state["mem_valid"].bool(), torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device)], -1)
            evidence = self.attention(query[:, None], keys, values, key_padding_mask=~valid, need_weights=False)[0][:, 0]
            # Empty memory has zero influence even if attention biases learn.
            use = prev_state["mem_valid"].any(-1, keepdim=True)
            if self.read_mode == "start_only":
                use = use & (prev_state["imag_steps"] == 0)
            gate = torch.sigmoid(self.read_gate(torch.cat([s_deter, z_deter, evidence], -1)))
            gated_delta = gate * self.read_out(evidence)
            retrieved = self.read_norm(s_deter + gated_delta)
            if collector is not None:
                diagnostic_use = use if diagnostic_rows is None else use & diagnostic_rows[:, None]
                collector.read(self, query, keys, valid,
                               prev_state["mem_age"] + prev_state["imag_steps"] + 1,
                               diagnostic_use, gate, s_deter, gated_delta, retrieved)
            s_deter = torch.where(use, retrieved, s_deter)
        state = self._merge_state(self._prior(self.s, s_deter, sample), self._prior(self.z, z_deter, sample))
        # No clone/write/refresh: the branch keeps the exact evidence snapshot.
        for key in ("mem_tokens", "mem_valid", "mem_age"):
            state[key] = prev_state[key]
        state["imag_steps"] = prev_state["imag_steps"] + 1
        return state

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True,
                 memory_tokens=None):
        if embed.ndim != 2 or embed.shape[-1] != self._embed:
            raise ValueError(f"Expected embed (batch, {self._embed})")
        batch = embed.shape[0]
        if is_first.shape != (batch,):
            raise ValueError("is_first must be (batch,)")
        if prev_state is None:
            prev_state = self.initial(batch)
            prev_action = embed.new_zeros(batch, self._num_actions)
        else:
            self._validate_state(prev_state)
            initial = self.initial(batch)
            first = is_first.bool()
            prev_state = {key: torch.where(first.reshape((batch,) + (1,) * (value.ndim - 1)), initial[key], value)
                          for key, value in prev_state.items()}
            if prev_action is None:
                raise ValueError("Previous real action is required when a carry exists")
            prev_action = torch.where(first[:, None], torch.zeros_like(prev_action), prev_action)
        prior = self.img_step(prev_state, prev_action, sample=sample)
        s_prior, z_prior = self._split_state(prior)
        posts = []
        for branch, branch_prior in ((self.s, s_prior), (self.z, z_prior)):
            hidden = branch._obs_out_layers(torch.cat([branch_prior["deter"], embed], -1))
            stats = branch._suff_stats_layer("obs", hidden)
            distribution = branch.get_dist(stats)
            posts.append(dict(deter=branch_prior["deter"], stoch=distribution.sample() if sample else distribution.mode(), **stats))
        post = self._merge_state(*posts)
        for key in ("mem_tokens", "mem_valid", "mem_age"):
            post[key] = prior[key]
        post["imag_steps"] = torch.zeros_like(prior["imag_steps"])
        if self.memory_enabled:
            if memory_tokens is None:
                raise ValueError("Received RGB observations require current image-only memory tokens")
            if memory_tokens.shape != (batch, self.tokens_per_observation, self.memory_dim):
                raise ValueError("Writer must return four tokens per current observation")
            n = self.tokens_per_observation
            post["mem_tokens"] = torch.cat([prior["mem_tokens"][:, n:], memory_tokens], 1)
            post["mem_valid"] = torch.cat([prior["mem_valid"][:, n:], torch.ones(batch, n, dtype=torch.bool, device=embed.device)], 1)
            post["mem_age"] = torch.cat([prior["mem_age"][:, n:] + 1, embed.new_zeros(batch, n)], 1)
        return post, prior

    def observe(self, embed, action, is_first, state=None, memory_tokens=None,
                sequence_valid=None, loss_mask=None):
        """Observe causal rows; padding is held and burn-in ends with detach.

        ``action[:, t]`` leads into ``image[:, t]``. The first learning row is
        inferred after detaching the history state, so no learning gradient
        propagates through preceding burn-in rows.
        """
        batch, length = embed.shape[:2]
        if length < 1:
            raise ValueError("Empty observation sequence")
        valid = torch.ones(batch, length, dtype=torch.bool, device=embed.device) if sequence_valid is None else sequence_valid.bool()
        if valid.shape != (batch, length):
            raise ValueError("sequence_valid must match batch,time")
        learn = valid if loss_mask is None else loss_mask.bool() & valid
        state = self.initial(batch) if state is None else state
        posts, priors = [], []
        for time in range(length):
            boundary = learn[:, time] & (~learn[:, time - 1] if time else torch.ones(batch, dtype=torch.bool, device=embed.device))
            state = {key: torch.where(boundary.reshape((batch,) + (1,) * (value.ndim - 1)), value.detach(), value)
                     for key, value in state.items()}
            indices = valid[:, time].nonzero(as_tuple=False).flatten()
            if indices.numel():
                previous = {key: value[indices] for key, value in state.items()}
                tokens = None if memory_tokens is None else memory_tokens[indices, time]
                new_post, new_prior = self.obs_step(previous, action[indices, time], embed[indices, time], is_first[indices, time], memory_tokens=tokens)
                post = {key: value.index_copy(0, indices, new_post[key]) for key, value in state.items()}
                prior = {key: value.index_copy(0, indices, new_prior[key]) for key, value in state.items()}
            else:
                post = state
                prior = state
            posts.append(post)
            priors.append(prior)
            state = post
        return ({key: torch.stack([row[key] for row in posts], 1) for key in state},
                {key: torch.stack([row[key] for row in priors], 1) for key in state})

    def s_feature(self, state):
        return torch.cat([state["stoch"][..., :self._stoch // 2, :].flatten(-2), state["deter"][..., :self._deter // 2]], -1)

