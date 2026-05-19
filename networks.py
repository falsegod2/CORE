import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
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
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions #去掉+ 1
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + self._num_actions # 统一使用 self._num_actions
            #inp_dim = self._stoch * self._discrete + num_actions + 1
        else:
            inp_dim = self._stoch + self._num_actions # 统一使用 self._num_actions
            #inp_dim = self._stoch + num_actions + 1
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}

        return post, prior
    '''彻底删除 observe_zoomed 方法
    def observe_zoomed(self, embed_zoomed, action_zoomed, is_first_zoomed, rely_post, rely_prior):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed_zoomed, action_zoomed, is_first_zoomed = swap(embed_zoomed), swap(action_zoomed), swap(is_first_zoomed)

        rely_post = {k: swap(v) for k, v in rely_post.items()}
        rely_prior = {k: swap(v) for k, v in rely_prior.items()}

        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post_zoomed, prior_zoomed = tools.static_scan_zoomed(
            lambda rely_state, prev_act, embed_zoomed, is_first_zoomed: self.obs_step(
                rely_state, prev_act, embed_zoomed, is_first_zoomed
            ),
            (action_zoomed, embed_zoomed, is_first_zoomed), 
            (rely_post, rely_prior),
        )

        post_zoomed = {k: swap(v) for k, v in post_zoomed.items()}
        prior_zoomed = {k: swap(v) for k, v in prior_zoomed.items()}

        return post_zoomed, prior_zoomed
    '''
    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        RSSM posterior update.

        修改重点：
        1. 去掉 prev_action *= ... 这种 inplace 操作。
        2. 不再原地修改 prev_state 字典，而是构造 new_prev_state。
        3. 保持原始逻辑：episode 第一帧使用 initial state 和 zero action。
        """

        batch_size = len(is_first)

        # is_first 可能是 bool，也可能是 float，这里统一成 bool 判断 reset。
        is_first = is_first.to(embed.device)
        is_first_bool = is_first.bool()

        if prev_state is None or torch.all(is_first_bool):
            prev_state = self.initial(batch_size)

            prev_action = torch.zeros(
                (batch_size, self._num_actions),
                device=self._device,
                dtype=embed.dtype,
            )

        elif torch.any(is_first_bool):
            # reset mask: [B, 1]
            reset = is_first_bool[:, None].to(embed.dtype)

            # 关键修正：
            # 原来是 prev_action *= 1.0 - is_first
            # 这里改成非原地操作，避免 autograd version mismatch。
            prev_action = prev_action.to(embed.device, dtype=embed.dtype)
            prev_action = prev_action * (1.0 - reset)

            init_state = self.initial(batch_size)

            new_prev_state = {}
            for key, val in prev_state.items():
                reset_r = torch.reshape(
                    reset,
                    reset.shape + (1,) * (len(val.shape) - len(reset.shape)),
                )

                init_val = init_state[key].to(device=val.device, dtype=val.dtype)

                # 不要写 prev_state[key] = ... 原地覆盖旧 state，
                # 构造一个新的 state 字典更安全。
                new_prev_state[key] = (
                    val * (1.0 - reset_r)
                    + init_val * reset_r
                )

            prev_state = new_prev_state

        else:
            # 没有 reset，也要确保 dtype/device 一致。
            prev_action = prev_action.to(embed.device, dtype=embed.dtype)

        prior = self.img_step(prev_state, prev_action)

        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)

        stats = self._suff_stats_layer("obs", x)

        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()

        post = {
            "stoch": stoch,
            "deter": prior["deter"],
            **stats,
        }

        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss

class SlotAttention(nn.Module):
    """Slot Attention with optional affordance bias."""

    def __init__(
        self,
        num_slots,
        token_dim,
        slot_dim=128,
        iters=3,
        hidden_dim=256,
        eps=1e-8,
        affordance_bias=1.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.affordance_bias = affordance_bias

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        self.norm_tokens = nn.LayerNorm(token_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(token_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(token_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.scale = slot_dim ** -0.5
        self.apply(tools.weight_init)

    def forward(self, tokens, affordance=None):
        # tokens: [B, N, token_dim]
        b, n, _ = tokens.shape

        tokens = self.norm_tokens(tokens)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        mu = self.slots_mu.expand(b, -1, -1)
        sigma = torch.exp(self.slots_logsigma).expand(b, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        if affordance is not None:
            affordance = affordance.reshape(b, 1, n).to(tokens.dtype)
            affordance = affordance.clamp(0.0, 1.0)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots

            q = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale

            # Slot Attention normalizes over slots first, so adding the same
            # token-wise affordance bias to every slot would cancel out. Instead,
            # we first compute slot competition and then reweight each slot
            # distribution over tokens toward high-affordance patches.
            attn = torch.softmax(logits, dim=1) + self.eps
            if affordance is not None and self.affordance_bias != 0:
                token_weight = 1.0 + self.affordance_bias * affordance
                attn = attn * token_weight
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )
            slots = slots.reshape(b, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn
    
class ObjectDynamics(nn.Module):
    """
    Dyn-O style object-level dynamics module.

    Input:
        slots:  [B, K, D]
        action: [B, A]
    Output:
        pred_next_slots: [B, K, D]
    """

    def __init__(
        self,
        slot_dim,
        action_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.action_proj = nn.Linear(action_dim, slot_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.interaction = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.pred = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.apply(tools.weight_init)

    def forward(self, slots, action):
        # slots: [B, K, D]
        # action: [B, A]
        action_token = self.action_proj(action).unsqueeze(1)

        # 把 action 作为一个额外 token，让 object slots 可以读取动作信息。
        x = torch.cat([slots, action_token], dim=1)

        # 对象之间做 self-attention，动作 token 也参与交互。
        x = self.interaction(x)

        # 只取前 K 个 object tokens，最后一个 action token 丢掉。
        obj_hidden = x[:, :-1]

        # residual prediction：预测 slot 的变化量，而不是完全重建 slot。
        delta = self.pred(obj_hidden)
        pred_next_slots = slots + delta
        return pred_next_slots
    

class ObjectSSMDynamics(nn.Module):
    """
    AGOC-Gate Object-RSSM branch.

    This module is intentionally not a set of K independent RSSMs. Instead, it
    follows the Dyn-O-style object-stream idea: every slot owns its own recurrent
    object state, but all slots share one transition/posterior/prior model. A
    lightweight interaction transformer lets object streams and the action token
    communicate before the shared RSSM transition.

    The branch operates only on a dynamic subspace of each slot. Static visual
    appearance and background context stay in the original AGOC-Gate embedding;
    this branch models action-conditioned object dynamics.

    Input:
        slots_seq:  [B, T, K, slot_dim]
        action_seq: [B, T, A]
    Output:
        pred_next_dyn: [B, T, K, dyn_dim]
        optional KL:   [B, T, K]
    """

    def __init__(
        self,
        slot_dim,
        action_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
        dyn_dim=64,
        min_std=0.1,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.dyn_dim = dyn_dim
        self.min_std = min_std

        # Dynamic subspace projection. Object-RSSM only models this subspace.
        self.dyn_encoder = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, dyn_dim),
            nn.SiLU(),
            nn.Linear(dyn_dim, dyn_dim),
        )

        self.action_proj = nn.Linear(action_dim, dyn_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dyn_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.interaction = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Shared deterministic transition for all object streams.
        self.rssm_cell = nn.GRUCell(dyn_dim, dyn_dim)

        # Prior p(z_t^k | h_t^k) and posterior q(z_t^k | h_t^k, x_t^k).
        self.prior_stats = nn.Sequential(
            nn.LayerNorm(dyn_dim),
            nn.Linear(dyn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dyn_dim),
        )
        self.post_stats = nn.Sequential(
            nn.LayerNorm(2 * dyn_dim),
            nn.Linear(2 * dyn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dyn_dim),
        )

        # Object feature used as context for the global RSSM and as the basis for
        # next-dynamic prediction.
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(2 * dyn_dim),
            nn.Linear(2 * dyn_dim, dyn_dim),
            nn.SiLU(),
            nn.Linear(dyn_dim, dyn_dim),
        )

        self.pred = nn.Sequential(
            nn.LayerNorm(dyn_dim),
            nn.Linear(dyn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dyn_dim),
        )

        self.apply(tools.weight_init)

    def encode_dynamic(self, slots):
        """Project full slots to the dynamic subspace.

        slots can be [B, T, K, D] or [B, K, D]. The output keeps the leading
        dimensions and replaces D by dyn_dim.
        """
        return self.dyn_encoder(slots)

    def _stats(self, raw):
        mean, std_raw = torch.chunk(raw, 2, dim=-1)
        std = F.softplus(std_raw + 0.5) + self.min_std
        return mean, std

    def _kl_normal(self, post_mean, post_std, prior_mean, prior_std):
        # KL(q || p), returned per slot: [B, K]
        var_ratio = (post_std / prior_std).pow(2)
        t1 = ((post_mean - prior_mean) / prior_std).pow(2)
        kl = 0.5 * (var_ratio + t1 - 1.0 - torch.log(var_ratio + 1e-8))
        return kl.mean(dim=-1)

    def _zero_state(self, batch_size, num_slots, device, dtype):
        h = torch.zeros(batch_size, num_slots, self.dyn_dim, device=device, dtype=dtype)
        z = torch.zeros(batch_size, num_slots, self.dyn_dim, device=device, dtype=dtype)
        return {"h": h, "z": z}

    def _normalize_state(self, state, batch_size, num_slots, device, dtype):
        if state is None:
            return self._zero_state(batch_size, num_slots, device, dtype)
        if isinstance(state, dict):
            h = state.get("h", None)
            z = state.get("z", None)
            if h is None:
                h = torch.zeros(batch_size, num_slots, self.dyn_dim, device=device, dtype=dtype)
            else:
                h = h.to(device=device, dtype=dtype)
            if z is None:
                z = torch.zeros_like(h)
            else:
                z = z.to(device=device, dtype=dtype)
            return {"h": h, "z": z}
        # Backward compatibility with v2 checkpoints where state was only h.
        h = state.to(device=device, dtype=dtype)
        z = torch.zeros_like(h)
        return {"h": h, "z": z}

    def _apply_reset(self, state, reset):
        if reset is None:
            return state
        reset = reset.float()
        if reset.ndim == 1:
            reset = reset[:, None, None]
        elif reset.ndim == 2:
            reset = reset[:, :, None]
        state = {
            "h": state["h"] * (1.0 - reset),
            "z": state["z"] * (1.0 - reset),
        }
        return state

    def _transition(self, dyn_i, action_i, state):
        """One shared Object-RSSM transition.

        Args:
            dyn_i:    [B, K, dyn_dim], current observed dynamic slot component.
            action_i: [B, A], action conditioning obs_t -> obs_{t+1}.
            state:    dict with h,z, each [B, K, dyn_dim].
        Returns:
            new_state, pred_next_dyn, kl_per_slot, object_feat
        """
        b, k, dd = dyn_i.shape
        h_prev, z_prev = state["h"], state["z"]

        action_token = self.action_proj(action_i).unsqueeze(1)  # [B, 1, dyn_dim]

        # Slots communicate through the previous stochastic object state and an
        # action token. This is the object-stream analogue of a shared RSSM prior.
        x = torch.cat([z_prev, action_token], dim=1)             # [B, K+1, dyn_dim]
        x = self.interaction(x)
        obj_input = x[:, :k]                                    # [B, K, dyn_dim]

        h = self.rssm_cell(
            obj_input.reshape(b * k, dd),
            h_prev.reshape(b * k, dd),
        ).reshape(b, k, dd)

        prior_mean, prior_std = self._stats(self.prior_stats(h))
        post_inp = torch.cat([h, dyn_i], dim=-1)
        post_mean, post_std = self._stats(self.post_stats(post_inp))

        # Use posterior mode for stability. The branch remains stochastic via KL,
        # but avoids adding sampling noise to the global RSSM input.
        z = post_mean
        kl = self._kl_normal(post_mean, post_std, prior_mean, prior_std)

        object_feat = self.feat_proj(torch.cat([h, z], dim=-1))
        delta = self.pred(object_feat)
        pred_next_dyn = dyn_i + delta

        new_state = {"h": h, "z": z}
        return new_state, pred_next_dyn, kl, object_feat

    def context_sequence(self, slots_seq, action_seq, tao_weights=None, is_first=None, init_state=None):
        """Run Object-RSSM over a sequence and return TAO-weighted context.

        Args:
            slots_seq:   [B, T, K, slot_dim]
            action_seq:  [B, T, A]
            tao_weights: [B, T, K] or None
            is_first:    [B, T] or [B, T, 1]
            init_state:  dict or None
        Returns:
            context_seq: [B, T, dyn_dim]
            final_state: dict with h,z
        """
        b, t, k, d = slots_seq.shape
        assert d == self.slot_dim, (d, self.slot_dim)

        dyn_seq = self.encode_dynamic(slots_seq)
        state = self._normalize_state(init_state, b, k, slots_seq.device, slots_seq.dtype)

        contexts = []
        for i in range(t):
            reset_i = is_first[:, i] if is_first is not None else None
            state = self._apply_reset(state, reset_i)
            state, _, _, object_feat = self._transition(dyn_seq[:, i], action_seq[:, i], state)

            if tao_weights is not None:
                w = tao_weights[:, i].detach().to(dtype=object_feat.dtype)
                context = torch.einsum("bk,bkd->bd", w, object_feat)
            else:
                context = object_feat.mean(dim=1)
            contexts.append(context)

        return torch.stack(contexts, dim=1), state

    def step_context(self, slots, action, state=None, tao_weights=None, is_first=None):
        """Online one-step Object-RSSM context update for Agent._policy()."""
        b, k, d = slots.shape
        assert d == self.slot_dim, (d, self.slot_dim)

        dyn = self.encode_dynamic(slots)
        state = self._normalize_state(state, b, k, slots.device, slots.dtype)
        state = self._apply_reset(state, is_first)
        state, _, _, object_feat = self._transition(dyn, action, state)

        if tao_weights is not None:
            w = tao_weights.detach().to(dtype=object_feat.dtype)
            context = torch.einsum("bk,bkd->bd", w, object_feat)
        else:
            context = object_feat.mean(dim=1)
        return context, state

    def forward(self, slots_seq, action_seq, is_first=None, return_kl=False):
        # slots_seq:  [B, T, K, slot_dim]
        # action_seq: [B, T, A]
        b, t, k, d = slots_seq.shape
        assert d == self.slot_dim, (d, self.slot_dim)

        dyn_seq = self.encode_dynamic(slots_seq)
        state = self._zero_state(b, k, slots_seq.device, slots_seq.dtype)
        preds = []
        kls = []

        for i in range(t):
            reset_i = is_first[:, i] if is_first is not None else None
            state = self._apply_reset(state, reset_i)
            state, pred_next_dyn, kl, _ = self._transition(dyn_seq[:, i], action_seq[:, i], state)
            preds.append(pred_next_dyn)
            kls.append(kl)

        pred_seq = torch.stack(preds, dim=1)  # [B, T, K, dyn_dim]
        if return_kl:
            return pred_seq, torch.stack(kls, dim=1)  # [B, T, K]
        return pred_seq

class ObjectCentricConvEncoder(nn.Module):
    """
    CNN -> spatial tokens -> affordance-biased Slot Attention -> flat Dreamer embedding.

    输出仍然是 [B, T, num_slots * slot_dim]，
    所以 RSSM、actor、critic 主体可以不改。
    """

    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        num_slots=16,
        slot_dim=128,
        slot_iters=3,
        affordance_bias=1.0,
        use_coords=True,
        task_embed_dim=512,
        tao_include_flat=True,
        tao_include_global=True,
        tao_include_affordance=True,
        tao_include_task=True,
        tao_affordance_weight=1.0,
        tao_task_weight=1.0,
        tao_temperature=5.0,
    ):
        super().__init__()
        act = getattr(torch.nn, act)

        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))

        in_dim = input_ch
        out_dim = depth
        layers = []

        for _ in range(stages):
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

        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        self.use_coords = use_coords
        token_dim = in_dim + (2 if use_coords else 0)

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            token_dim=token_dim,
            slot_dim=slot_dim,
            iters=slot_iters,
            hidden_dim=max(slot_dim * 2, 256),
            affordance_bias=affordance_bias,
        )

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.task_embed_dim = task_embed_dim
        self.tao_include_flat = tao_include_flat
        self.tao_include_global = tao_include_global
        self.tao_include_affordance = tao_include_affordance
        self.tao_include_task = tao_include_task
        self.tao_affordance_weight = tao_affordance_weight
        self.tao_task_weight = tao_task_weight
        self.tao_temperature = tao_temperature

        self.slot_task_proj = nn.Linear(slot_dim, task_embed_dim)
        self.slot_task_proj.apply(tools.weight_init)

        outdim = 0
        if tao_include_flat:
            outdim += num_slots * slot_dim
        if tao_include_global:
            outdim += slot_dim
        if tao_include_affordance:
            outdim += slot_dim + num_slots
        if tao_include_task:
            outdim += slot_dim + num_slots
        self.outdim = outdim

        self.last_attn = None
        self.last_affordance = None
        self.last_aff_scores = None
        self.last_aff_scores_norm = None
        self.last_task_scores = None
        self.last_tao_weights = None
        self.last_slots = None
        self.last_lead_shape = None

    def _coords(self, b, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)
        return coords.expand(b, -1, -1)

    def forward(self, obs, affordance=None, task_embed=None):
        # obs: [B, T, H, W, C]
        obs = obs - 0.5
        lead_shape = list(obs.shape[:-3])

        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)

        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        if self.use_coords:
            coords = self._coords(b, h, w, x.device, x.dtype)
            tokens = torch.cat([tokens, coords], dim=-1)

        aff = None
        if affordance is not None:
            aff = affordance.reshape((-1,) + tuple(affordance.shape[-3:]))
            aff = aff.permute(0, 3, 1, 2)
            aff = F.interpolate(aff, size=(h, w), mode="bilinear", align_corners=False)
            aff = aff.reshape(b, h * w)

        slots, attn = self.slot_attention(tokens, aff)
        self.last_slots = slots
        self.last_lead_shape = lead_shape
        self.last_attn = attn
        self.last_affordance = aff


        # Slot-level affordance score: how much each object slot attends to
        # high-affordance image patches. Shape: [B*T, K].
        if aff is not None:
            aff_scores = torch.einsum("bkn,bn->bk", attn, aff.clamp(0.0, 1.0))
        else:
            aff_scores = torch.zeros(b, self.num_slots, device=slots.device, dtype=slots.dtype)

        # 关键修改：TAO 关心的是同一帧内不同 slots 的相对差异。
        # 原始 aff_scores 的 slot-wise 差距太小，softmax 会一直接近均匀。
        aff_mean = aff_scores.mean(dim=-1, keepdim=True)
        aff_std = aff_scores.std(dim=-1, keepdim=True, unbiased=False)
        aff_scores_norm = (aff_scores - aff_mean) / (aff_std + 1e-4)
        aff_scores_norm = aff_scores_norm.clamp(-5.0, 5.0)

        # Slot-level MineCLIP/task relevance. The environment wrapper can provide
        # the prompt embedding cached by the affordance/MineCLIP module.
        if task_embed is not None:
            task = task_embed.reshape((-1,) + tuple(task_embed.shape[-1:])).to(slots.dtype)
            task = F.normalize(task, dim=-1)
            slot_sem = F.normalize(self.slot_task_proj(slots), dim=-1)
            task_scores = torch.einsum("bkd,bd->bk", slot_sem, task)
        else:
            task_scores = torch.zeros_like(aff_scores_norm)

        tao_logits = (
            self.tao_affordance_weight * aff_scores_norm
            + self.tao_task_weight * task_scores
        ) * self.tao_temperature

        tao_weights = torch.softmax(tao_logits, dim=-1)

        global_slot = slots.mean(dim=1)
        affordance_slot = torch.einsum("bk,bkd->bd", tao_weights, slots)
        task_slot = affordance_slot if not self.tao_include_task else torch.einsum(
            "bk,bkd->bd", tao_weights, slots
        )


        pieces = []

        if self.tao_include_flat:
            # Residual TAO-gated flat slots.
            # 保留完整 slots，但让 TAO 认为重要的 slots 在 RSSM 输入中更突出。
            #
            # tao_weights: [B, K]
            # slots:       [B, K, D]
            #
            # 乘以 num_slots 是为了保持平均尺度：
            # 如果 tao_weights 完全均匀，slot_gate = 1，不改变原始 flat slots。
            slot_gate = self.num_slots * tao_weights.detach()
            slot_gate = slot_gate.clamp(0.0, 3.0)

            # 残差门控，避免低权重 slots 被完全压死。
            gate_strength = 0.5
            slot_gate = (1.0 - gate_strength) + gate_strength * slot_gate

            weighted_slots = slots * slot_gate.unsqueeze(-1)
            pieces.append(weighted_slots.reshape(b, self.num_slots * self.slot_dim))

        if self.tao_include_global:
            pieces.append(global_slot)

        if self.tao_include_affordance:
            pieces.extend([affordance_slot, aff_scores_norm])

        if self.tao_include_task:
            pieces.extend([task_slot, task_scores])

        out = torch.cat(pieces, dim=-1)
        out = out.reshape(lead_shape + [self.outdim])

        self.last_attn_vis = attn.detach()
        self.last_aff_vis = aff.detach() if aff is not None else None
        self.last_aff_scores = aff_scores
        self.last_aff_scores_norm = aff_scores_norm
        self.last_task_scores = task_scores
        self.last_tao_weights = tao_weights

        if not hasattr(self, "_debug_printed"):
            print("[ObjectCentricConvEncoder] slots:", slots.shape)
            print("[ObjectCentricConvEncoder] affordance:", None if aff is None else aff.shape)
            print("[ObjectCentricConvEncoder] task_embed:", None if task_embed is None else task_embed.shape)
            print("[ObjectCentricConvEncoder] attn:", attn.shape)
            print("[ObjectCentricConvEncoder] out:", out.shape)
            self._debug_printed = True

        return out

    def object_aux_losses(self):
        losses = {}
        align = self.affordance_alignment_loss()
        if align is not None:
            losses["affordance_align"] = align

        if self.last_attn is not None:
            # Penalize overlapping attention maps across slots to avoid all slots
            # collapsing onto the same high-affordance region.
            attn = F.normalize(self.last_attn, p=2, dim=-1)
            sim = torch.einsum("bkn,bmn->bkm", attn, attn)
            eye = torch.eye(self.num_slots, device=sim.device, dtype=torch.bool).unsqueeze(0)
            losses["slot_diversity"] = sim.masked_fill(eye, 0.0).sum(dim=(1, 2)) / max(self.num_slots * (self.num_slots - 1), 1)

        if self.last_tao_weights is not None:
            # Reportable regularizer: encourages non-degenerate task-affordance
            # object selection. Use a small scale in config.
            w = self.last_tao_weights.clamp_min(1e-8)
            losses["tao_entropy"] = -(w * torch.log(w)).sum(dim=-1)
        return losses

    def affordance_alignment_loss(self):
        """
        Align TAO-selected slot attention coverage with affordance map.

        Compared with all-slot coverage alignment, this version only requires
        the slots selected by TAO to cover high-affordance regions.
        This is more consistent with AGOC-Dyn / AGOC-Dyn-SSM.
        """
        attn = self.last_attn
        aff = self.last_affordance

        if attn is None or aff is None:
            return None

        # attn: [B, K, N]
        # aff:  [B, N]
        attn = attn.float()
        aff = aff.float().clamp(0.0, 1.0)

        # 如果 heatmap 全 0，避免归一化出问题。
        aff_sum = aff.sum(dim=-1, keepdim=True)

        # 轻微 sharpen affordance target。
        # 如果不 sharpen，4x4 heatmap 很容易接近均匀，loss 会长期接近 log(16)。
        aff_target = aff ** 2
        aff_target = aff_target / (aff_target.sum(dim=-1, keepdim=True) + 1e-8)

        # 如果某些样本 heatmap 近似全 0，则退化为原始 aff 的安全归一化。
        aff_safe = aff / (aff_sum + 1e-8)
        aff_target = torch.where(
            aff_sum > 1e-6,
            aff_target,
            torch.ones_like(aff_target) / aff_target.shape[-1],
        )

        tao_weights = getattr(self, "last_tao_weights", None)

        if tao_weights is not None:
            # tao_weights: [B, K]
            # detach 是为了让这个 loss 主要训练 attention 覆盖，
            # 而不是通过改变 tao_weights 来投机降低 loss。
            w = tao_weights.detach().float()

            # TAO-selected coverage: [B, N]
            slot_cover = torch.einsum("bk,bkn->bn", w, attn)
        else:
            # fallback: 如果没有 TAO，就用所有 slots 的平均 coverage。
            slot_cover = attn.mean(dim=1)

        slot_cover = slot_cover.clamp_min(1e-8)
        slot_cover = slot_cover / (slot_cover.sum(dim=-1, keepdim=True) + 1e-8)

        loss = -(aff_target * torch.log(slot_cover + 1e-8)).sum(dim=-1)

        # 如果 heatmap 全 0，这个 loss 没有意义，直接置 0。
        valid = (aff_sum.squeeze(-1) > 1e-6).float()
        loss = loss * valid

        return loss


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
        symlog_inputs=False,
        use_slots=False,
        num_slots=16,
        slot_dim=128,
        slot_iters=3,
        affordance_keys="heatmap",
        task_embed_keys="task_embed",
        affordance_bias=1.0,
        slot_use_coords=True,
        task_embed_dim=512,
        tao_include_flat=True,
        tao_include_global=True,
        tao_include_affordance=True,
        tao_include_task=True,
        tao_affordance_weight=1.0,
        tao_task_weight=1.0,
        tao_temperature=5.0,
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
        print(
                "Encoder Affordance shapes:",
                self.affordance_shapes if hasattr(self, "affordance_shapes") else {}
            )
        self.outdim = 0
        self.use_slots = use_slots
        self.affordance_keys = [
            k for k, v in shapes.items()
            if len(v) == 3 and re.match(affordance_keys, k)
        ]
        self.task_embed_keys = [
            k for k, v in shapes.items()
            if len(v) == 1 and re.match(task_embed_keys, k)
        ]

        self.affordance_shapes = {
            k: shapes[k] for k in self.affordance_keys
        }

        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)

            if use_slots:
                self._cnn = ObjectCentricConvEncoder(
                    input_shape,
                    cnn_depth,
                    act,
                    norm,
                    kernel_size,
                    minres,
                    num_slots=num_slots,
                    slot_dim=slot_dim,
                    slot_iters=slot_iters,
                    affordance_bias=affordance_bias,
                    use_coords=slot_use_coords,
                    task_embed_dim=task_embed_dim,
                    tao_include_flat=tao_include_flat,
                    tao_include_global=tao_include_global,
                    tao_include_affordance=tao_include_affordance,
                    tao_include_task=tao_include_task,
                    tao_affordance_weight=tao_affordance_weight,
                    tao_task_weight=tao_task_weight,
                    tao_temperature=tao_temperature,
                )
            else:
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

            if self.use_slots:
                affordance = None
                for key in self.affordance_keys:
                    if key in obs:
                        affordance = obs[key]
                        break
                task_embed = None
                for key in self.task_embed_keys:
                    if key in obs:
                        task_embed = obs[key]
                        break
                outputs.append(self._cnn(inputs, affordance, task_embed))
            else:
                outputs.append(self._cnn(inputs))

        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


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

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


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
