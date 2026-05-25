import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


import torch
from torch import nn
from torch import distributions as torchd
import tools
import numpy as np

class RSSM(nn.Module):
    def __init__(
        self, stoch=30, deter=200, hidden=200, rec_depth=1, discrete=False,
        act="SiLU", norm=True, mean_act="none", std_act="softplus", min_std=0.1,
        unimix_ratio=0.01, initial="learned", num_actions=None, embed=None, device=None,
        object_context_dim=0,
    ):
        super(RSSM, self).__init__()
        # 1. 维度拆分：s(受控), z(非受控)
        self._stoch_s = stoch // 2
        self._stoch_z = stoch - self._stoch_s
        self._deter_s = deter // 2
        self._deter_z = deter - self._deter_s
        
        self._stoch, self._deter = stoch, deter
        self._hidden, self._min_std = hidden, min_std
        self._rec_depth, self._discrete = rec_depth, discrete
        act_fn = getattr(torch.nn, act)
        self._mean_act, self._std_act = mean_act, std_act
        self._unimix_ratio, self._initial = unimix_ratio, initial
        self._num_actions = num_actions + 1
        self._embed, self._device = embed, device
        # CORE4-Full-NoDense: the encoder can append heatmap top-k object context
        # to the end of the embedding. The posterior s branch sees the full embedding,
        # while the z branch receives only the base visual/MLP embedding. This keeps
        # task-relevant affordance-object information in the controllable branch.
        self._object_context_dim = int(object_context_dim or 0)
        self._embed_base = int(embed) - self._object_context_dim
        if self._embed_base <= 0:
            self._embed_base = int(embed)
            self._object_context_dim = 0

        # 2. 受控分支网络 (输入包含动作)
        stoch_size_s = self._stoch_s * (self._discrete if self._discrete else 1)
        self._img_in_s = self._make_layer(stoch_size_s + self._num_actions, norm, act_fn)
        self._cell_s = GRUCell(self._hidden, self._deter_s, norm=norm)
        self._img_out_s = self._make_layer(self._deter_s, norm, act_fn)
        self._obs_out_s = self._make_layer(self._deter_s + self._embed, norm, act_fn)

        # 3. 非受控分支网络 (动作无关)
        stoch_size_z = self._stoch_z * (self._discrete if self._discrete else 1)
        self._img_in_z = self._make_layer(stoch_size_z, norm, act_fn)
        self._cell_z = GRUCell(self._hidden, self._deter_z, norm=norm)
        self._img_out_z = self._make_layer(self._deter_z, norm, act_fn)
        self._obs_out_z = self._make_layer(self._deter_z + self._embed_base, norm, act_fn)

        # 4. 映射层与逆动力学
        self._stat_s_img = self._make_stat_layer(self._stoch_s)
        self._stat_s_obs = self._make_stat_layer(self._stoch_s)
        self._stat_z_img = self._make_stat_layer(self._stoch_z)
        self._stat_z_obs = self._make_stat_layer(self._stoch_z)
        self._inverse_dynamics = nn.Sequential(
            nn.Linear(self._deter_s * 2, self._hidden), act_fn(),
            nn.Linear(self._hidden, num_actions)
        )

        # CORE3: interaction gate.
        # It reads only controllable-branch features and predicts whether the
        # current state is near a task-relevant interaction event.
        self._interaction_gate = nn.Sequential(
            nn.Linear(self._deter_s + stoch_size_s, self._hidden),
            nn.LayerNorm(self._hidden, eps=1e-03) if norm else nn.Identity(),
            act_fn(),
            nn.Linear(self._hidden, 1),
            nn.Sigmoid(),
        )
        self._interaction_gate.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(torch.zeros((1, self._deter), device=torch.device(self._device)), requires_grad=True)
        
    def get_interaction_score(self, state):
        """Predict an interaction probability from the controllable branch.

        CORE3-no-intrinsic-short removes the long-distance/jumpy branch, so this
        score is not used to gate jumpy imagination. It is still trained as an
        auxiliary representation signal: states around reward changes should be
        distinguishable in the controllable branch.
        """
        s_stoch = state["stoch_s"]
        if self._discrete:
            s_stoch = s_stoch.reshape(
                list(s_stoch.shape[:-2]) + [self._stoch_s * self._discrete]
            )
        feat_s = torch.cat([s_stoch, state["deter_s"]], -1)
        return self._interaction_gate(feat_s)

    def _make_layer(self, inp_dim, norm, act_fn):
        layers = [nn.Linear(inp_dim, self._hidden, bias=False)]
        if norm: layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        layers.append(act_fn()); net = nn.Sequential(*layers)
        net.apply(tools.weight_init); return net

    def _make_stat_layer(self, dim):
        l = nn.Linear(self._hidden, dim * (self._discrete if self._discrete else 2))
        l.apply(tools.uniform_weight_init(1.0)); return l

    def initial(self, batch_size):
        # 1. 初始化确定性状态 (deter)
        deter_s = torch.zeros(batch_size, self._deter_s).to(self._device)
        deter_z = torch.zeros(batch_size, self._deter_z).to(self._device)
        
        if self._initial == "learned":
            W_s, W_z = torch.split(torch.tanh(self.W), [self._deter_s, self._deter_z], -1)
            deter_s = W_s.repeat(batch_size, 1)
            deter_z = W_z.repeat(batch_size, 1)

        # 2. 构造初始状态字典
        state = {"deter_s": deter_s, "deter_z": deter_z}
        
        # --- 修复点：获取受控分支(s)的初始随机状态及分布参数 ---
        stoch_s, stats_s = self._get_stoch_and_stats_init(deter_s, "s")
        state["stoch_s"] = stoch_s
        state.update({f"s_{k}": v for k, v in stats_s.items()}) # 添加 s_logit 或 s_mean/std
        
        # --- 修复点：获取非受控分支(z)的初始随机状态及分布参数 ---
        stoch_z, stats_z = self._get_stoch_and_stats_init(deter_z, "z")
        state["stoch_z"] = stoch_z
        state.update({f"z_{k}": v for k, v in stats_z.items()}) # 添加 z_logit 或 z_mean/std
        
        return state

    # --- 辅助函数：统一获取初始随机态和参数 ---
    def _get_stoch_and_stats_init(self, deter, branch):
        net = self._img_out_s if branch == "s" else self._img_out_z
        stat_layer = self._stat_s_img if branch == "s" else self._stat_z_img
        stoch_dim = self._stoch_s if branch == "s" else self._stoch_z
        
        x = net(deter)
        stats = self._suff_stats_layer(stat_layer, x, stoch_dim)
        stoch = self.get_dist(stats).mode()
        return stoch, stats

    # --- 必须保留的核心接口：处理正常序列 ---
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(prev_state[0], prev_act, embed, is_first),
            (action, embed, is_first), (state, state),
        )
        return {k: swap(v) for k, v in post.items()}, {k: swap(v) for k, v in prior.items()}

    # --- 必须保留的核心接口：处理缩放跳跃序列 ---
    def observe_zoomed(self, embed_z, action_z, is_f_z, rely_post, rely_prior):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed_z, action_z, is_f_z = swap(embed_z), swap(action_z), swap(is_f_z)
        rely_p = {k: swap(v) for k, v in rely_post.items()}
        rely_pr = {k: swap(v) for k, v in rely_prior.items()}
        # 注意：这里调用的是重构后的 obs_step，它会自动处理解耦状态
        post_z, prior_z = tools.static_scan_zoomed(
            lambda r_s, p_a, e_z, i_f_z: self.obs_step(r_s, p_a, e_z, i_f_z),
            (action_z, embed_z, is_f_z), (rely_p, rely_pr),
        )
        return {k: swap(v) for k, v in post_z.items()}, {k: swap(v) for k, v in prior_z.items()}

    # --- 必须保留的核心接口：闭眼想象 ---
    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)[0]
        return {k: swap(v) for k, v in prior.items()}

    def _split_embed_for_branches(self, embed):
        if self._object_context_dim > 0 and embed.shape[-1] > self._object_context_dim:
            embed_base = embed[..., : -self._object_context_dim]
            embed_full = embed
        else:
            embed_base = embed
            embed_full = embed
        return embed_full, embed_base

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(self._device)
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_s = self.initial(len(is_first))
            for k, v in prev_state.items():
                is_f_r = torch.reshape(is_first, is_first.shape + (1,) * (len(v.shape) - len(is_first.shape)))
                prev_state[k] = v * (1.0 - is_f_r) + init_s[k] * is_f_r

        # --- 核心修复：自动补全动作维度 ---
        # 如果传入的是原始动作(12维)，自动补一个0(非跳跃标志)，变成13维
        if prev_action is not None and prev_action.shape[-1] == self._num_actions - 1:
            prev_action = torch.cat([prev_action, torch.zeros_like(prev_action[..., :1])], -1)

        prior = self.img_step(prev_state, prev_action)
        embed_s, embed_z = self._split_embed_for_branches(embed)
        # 受控后验：读取完整 embedding，其中最后几维可以是 affordance top-k object context。
        stats_s = self._suff_stats_layer(self._stat_s_obs, self._obs_out_s(torch.cat([prior["deter_s"], embed_s], -1)), self._stoch_s)
        stoch_s = self.get_dist(stats_s).sample() if sample else self.get_dist(stats_s).mode()
        # 非受控后验：不读取 affordance-object context，避免任务相关局部对象信息泄露到 z 分支。
        stats_z = self._suff_stats_layer(self._stat_z_obs, self._obs_out_z(torch.cat([prior["deter_z"], embed_z], -1)), self._stoch_z)
        stoch_z = self.get_dist(stats_z).sample() if sample else self.get_dist(stats_z).mode()

        post = {"stoch_s": stoch_s, "deter_s": prior["deter_s"], "stoch_z": stoch_z, "deter_z": prior["deter_z"]}
        # 合并分布参数以便后续计算 KL (添加前缀防止冲突)
        post.update({f"s_{k}": v for k, v in stats_s.items()})
        post.update({f"z_{k}": v for k, v in stats_z.items()})
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # --- 受控分支演化 (s) ---
        prev_s = prev_state["stoch_s"]
        if self._discrete:
            # 修复点：显式指定维度
            s_dim = self._stoch_s * self._discrete
            if prev_s.numel() > 0:
                prev_s = prev_s.reshape(list(prev_s.shape[:-2]) + [s_dim])
        
        # 确保 prev_action 与 prev_s 形状对齐（处理空 Batch 情况）
        x_s = torch.cat([prev_s, prev_action], -1)
        x_s, deter_s = self._cell_s(self._img_in_s(x_s), [prev_state["deter_s"]])
        stats_s = self._suff_stats_layer(self._stat_s_img, self._img_out_s(x_s), self._stoch_s)
        stoch_s = self.get_dist(stats_s).sample() if sample else self.get_dist(stats_s).mode()

        # --- 非受控分支演化 (z) ---
        prev_z = prev_state["stoch_z"]
        if self._discrete:
            z_dim = self._stoch_z * self._discrete
            if prev_z.numel() > 0:
                prev_z = prev_z.reshape(list(prev_z.shape[:-2]) + [z_dim])
            
        x_z, deter_z = self._cell_z(self._img_in_z(prev_z), [prev_state["deter_z"]])
        stats_z = self._suff_stats_layer(self._stat_z_img, self._img_out_z(x_z), self._stoch_z)
        stoch_z = self.get_dist(stats_z).sample() if sample else self.get_dist(stats_z).mode()

        prior = {"stoch_s": stoch_s, "deter_s": deter_s[0], "stoch_z": stoch_z, "deter_z": deter_z[0]}
        prior.update({f"s_{k}": v for k, v in stats_s.items()})
        prior.update({f"z_{k}": v for k, v in stats_z.items()})
        return prior

    def _suff_stats_layer(self, layer, x, dim):
        x = layer(x)
        if self._discrete: return {"logit": x.reshape(list(x.shape[:-1]) + [dim, self._discrete])}
        mean, std = torch.split(x, [dim] * 2, -1)
        mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
        std = {"softplus": lambda: torch.softplus(std), "abs": lambda: torch.abs(std + 1)}[self._std_act]()
        return {"mean": mean, "std": std + self._min_std}

    def get_dist(self, stats):
        if self._discrete:
            return torchd.independent.Independent(tools.OneHotDist(stats["logit"], unimix_ratio=self._unimix_ratio), 1)
        return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(stats["mean"], stats["std"]), 1))

    def get_feat(self, state):
        s_stoch = state.get("stoch_s", torch.tensor([]).to(self._device))
        z_stoch = state.get("stoch_z", torch.tensor([]).to(self._device))
        deter_s = state.get("deter_s", torch.tensor([]).to(self._device))
        deter_z = state.get("deter_z", torch.tensor([]).to(self._device))

        if self._discrete:
            s_dim = self._stoch_s * self._discrete
            z_dim = self._stoch_z * self._discrete
            if s_stoch.numel() > 0:
                s_stoch = s_stoch.reshape(list(s_stoch.shape[:-2]) + [s_dim])
            if z_stoch.numel() > 0:
                z_stoch = z_stoch.reshape(list(z_stoch.shape[:-2]) + [z_dim])
        
        # 确认顺序：s_stoch -> deter_s -> z_stoch -> deter_z
        # 这保证了前 (s_stoch + deter_s) 位全是受控信息
        return torch.cat([s_stoch, deter_s, z_stoch, deter_z], -1)

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        # 适配双分支的 KL 损失计算
        def get_branch_dist(state, prefix):
            # 从合并后的 state 字典中提取对应分支的参数
            branch_stats = {k[2:]: v for k, v in state.items() if k.startswith(f"{prefix}_")}
            return self.get_dist(branch_stats)

        kld = torchd.kl.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # 受控分支与非受控分支分别计算 KL
        kl_s = kld(get_branch_dist(post, "s"), get_branch_dist(sg(prior), "s"))
        kl_z = kld(get_branch_dist(post, "z"), get_branch_dist(sg(prior), "z"))
        
        # 按照 DreamerV3 逻辑计算 rep_loss 和 dyn_loss
        rep_loss = torch.clip(kl_s + kl_z, min=free)
        
        dyn_kl_s = kld(get_branch_dist(sg(post), "s"), get_branch_dist(prior, "s"))
        dyn_kl_z = kld(get_branch_dist(sg(post), "z"), get_branch_dist(prior, "z"))
        dyn_loss = torch.clip(dyn_kl_s + dyn_kl_z, min=free)

        return dyn_scale * dyn_loss + rep_scale * rep_loss, (kl_s + kl_z), dyn_loss, rep_loss



class AffordanceTopKObjectPool(nn.Module):
    """Lightweight object-token extractor guided by affordance heatmaps.

    This is the practical, no-extra-reward version of the Dyn-O / OC-STORM idea:
    it does not call SAM/Cutie online and does not create dense rewards. Instead,
    it uses the already available LS-Imagine heatmap as a spatial prior, extracts
    top-k task-relevant local CNN features, converts them into object tokens, and
    returns a compact context vector appended to the encoder embedding.
    """

    def __init__(self, in_channels, token_dim=128, context_dim=128, topk=4,
                 act="SiLU", norm=True, score_temp=0.2):
        super().__init__()
        act_fn = getattr(torch.nn, act)
        self._topk = int(topk)
        self._score_temp = float(score_temp)
        self._token_dim = int(token_dim)
        self._context_dim = int(context_dim)
        self._token_mlp = nn.Sequential(
            nn.Linear(in_channels + 3, token_dim, bias=False),
            nn.LayerNorm(token_dim, eps=1e-03) if norm else nn.Identity(),
            act_fn(),
            nn.Linear(token_dim, token_dim, bias=False),
            nn.LayerNorm(token_dim, eps=1e-03) if norm else nn.Identity(),
            act_fn(),
        )
        heads = 4 if token_dim % 4 == 0 else 1
        self._self_attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self._context_mlp = nn.Sequential(
            nn.Linear(token_dim, context_dim, bias=False),
            nn.LayerNorm(context_dim, eps=1e-03) if norm else nn.Identity(),
            act_fn(),
        )
        self.apply(tools.weight_init)

    def forward(self, spatial_feat, heatmap):
        # spatial_feat: [B, T, C, H, W], heatmap: [B, T, H0, W0, 1] or [B,T,H0,W0]
        b, t, c, h, w = spatial_feat.shape
        bt = b * t
        x = spatial_feat.reshape(bt, c, h, w)
        if heatmap is None:
            hm = torch.zeros(bt, 1, h, w, device=x.device, dtype=x.dtype)
        else:
            hm = heatmap
            if hm.ndim == 5:
                hm = hm.reshape(bt, hm.shape[-3], hm.shape[-2], hm.shape[-1]).permute(0, 3, 1, 2)
            elif hm.ndim == 4:
                hm = hm.reshape(bt, 1, hm.shape[-2], hm.shape[-1])
            hm = hm.to(device=x.device, dtype=x.dtype)
            hm = F.interpolate(hm, size=(h, w), mode="bilinear", align_corners=False)
        scores = hm.flatten(1)  # [BT, HW]
        k = max(1, min(self._topk, scores.shape[-1]))
        vals, idx = torch.topk(scores, k=k, dim=-1)
        feat_flat = x.flatten(2).transpose(1, 2)  # [BT, HW, C]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, c)
        local_feat = torch.gather(feat_flat, 1, gather_idx)  # [BT, K, C]

        # normalized spatial coordinates and heatmap confidence per selected token
        yy = (idx // w).to(x.dtype) / max(1, h - 1)
        xx = (idx % w).to(x.dtype) / max(1, w - 1)
        token_in = torch.cat([local_feat, vals.unsqueeze(-1), yy.unsqueeze(-1), xx.unsqueeze(-1)], dim=-1)
        tokens = self._token_mlp(token_in)
        tokens_attn, _ = self._self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + tokens_attn
        weights = torch.softmax(vals / max(self._score_temp, 1e-6), dim=-1).unsqueeze(-1)
        pooled = torch.sum(tokens * weights, dim=1)
        context = self._context_mlp(pooled).reshape(b, t, self._context_dim)
        stats = {
            "aff_object_score_mean": vals.mean().detach(),
            "aff_object_score_std": vals.std().detach(),
            "aff_object_score_max": vals.max().detach(),
            "aff_object_context_norm": context.norm(dim=-1).mean().detach(),
            "aff_object_token_norm": tokens.norm(dim=-1).mean().detach(),
        }
        return context, stats


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
        use_aff_object_tokens=False,
        aff_object_keys="heatmap",
        aff_object_num=4,
        aff_object_token_dim=128,
        aff_object_context_dim=128,
        aff_object_score_temp=0.2,
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
        self._use_aff_object_tokens = False
        self._aff_object_keys = aff_object_keys
        self._last_aff_object_stats = {}
        self._aff_object_pool = None
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
            self._use_aff_object_tokens = bool(use_aff_object_tokens)
            self._aff_object_keys = aff_object_keys
            self._last_aff_object_stats = {}
            if self._use_aff_object_tokens:
                self._aff_object_pool = AffordanceTopKObjectPool(
                    self._cnn.out_channels,
                    token_dim=aff_object_token_dim,
                    context_dim=aff_object_context_dim,
                    topk=aff_object_num,
                    act=act,
                    norm=norm,
                    score_temp=aff_object_score_temp,
                )
                self.outdim += int(aff_object_context_dim)
            else:
                self._aff_object_pool = None
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
        obj_context = None
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            if self._use_aff_object_tokens and self._aff_object_pool is not None:
                cnn_embed, spatial = self._cnn(inputs, return_spatial=True)
                outputs.append(cnn_embed)
                heatmap = None
                for key in str(self._aff_object_keys).split("|"):
                    if key in obs:
                        heatmap = obs[key]
                        break
                obj_context, obj_stats = self._aff_object_pool(spatial, heatmap)
                self._last_aff_object_stats = obj_stats
            else:
                outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        # Important: object context is appended at the end so RSSM can route the
        # last aff_object_context_dim dimensions only to the controllable branch.
        if obj_context is not None:
            outputs.append(obj_context)
        outputs = torch.cat(outputs, -1)
        return outputs

    def get_extra_metrics(self):
        return getattr(self, "_last_aff_object_stats", {})


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

        self.out_channels = out_dim // 2
        self.outdim = self.out_channels * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs, return_spatial=False):
        obs = obs - 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        spatial = x.reshape(list(obs.shape[:-3]) + list(x.shape[1:]))
        flat = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        flat = flat.reshape(list(obs.shape[:-3]) + [flat.shape[-1]])
        if return_spatial:
            return flat, spatial
        return flat


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
