# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import math
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from affordance_map.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys, MultimodalSwinTransformerSys

import networks
import tools
import s_multistep_consistency

logger = logging.getLogger(__name__)
to_np = lambda x: x.detach().cpu().numpy()


def gradient_reverse(x, scale=1.0):
    """Forward identity with a compile-friendly negative backward scale."""
    scale = float(scale)
    return (1.0 + scale) * x.detach() - scale * x

class RewardEMA:
    """
    运行时的奖励均值和标准差估计 (Exponential Moving Average)
    用于标准化奖励信号，使训练更稳定
    """
    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class MCUnet(nn.Module):
    """多模态 U-Net，用于处理图像和文本特征"""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MCUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = MultimodalSwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                text_feature_dim=config.MODEL.TEXT_FEATURE_DIM,
                                heads=config.MODEL.HEADS)

    def forward(self, x, p):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x, p)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        if "heatmap" in getattr(self.encoder, "cnn_shapes", {}):
            raise ValueError(
                "RGB-only ISO variant forbids heatmap in the observation encoder. "
                "Keep heatmap only as an auxiliary prediction target."
            )
        self.embed_size = self.encoder.outdim
        
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act,
            config.norm, config.dyn_mean_act, config.dyn_std_act,
            config.dyn_min_std, config.unimix_ratio, config.initial,
            config.num_actions, self.embed_size, config.device,
        )

        self.heads = nn.ModuleDict()

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        self.heads["decoder"] = networks.MultiDecoder(feat_size, shapes, **config.decoder)

        self.heads["reward"] = networks.MLP(
            feat_size, (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"], config.units, config.act, config.norm,
            dist=config.reward_head["dist"], outscale=config.reward_head["outscale"],
            device=config.device, name="Reward",
        )

        self.heads["end"] = networks.MLP(
            feat_size, (), config.end_head["layers"], config.units, config.act, config.norm,
            dist="binary", outscale=config.end_head["outscale"], device=config.device, name="End",
        )

        self.heads["intrinsic"] = networks.MLP(
            feat_size, (255,) if config.intrinsic_head["dist"] == "symlog_disc" else (),
            config.intrinsic_head["layers"], config.units, config.act, config.norm,
            dist=config.intrinsic_head["dist"], outscale=config.intrinsic_head["outscale"],
            device=config.device, name="Intrinsic",
        )

        # Diagnostic/adversarial action probe for the action-free Z branch.
        # Gradient reversal trains the probe to recognize actions while pushing Z
        # toward action invariance. The scale can be set to zero for ablation.
        act_fn = getattr(torch.nn, config.act)
        z_stoch_dim = self.dynamics._stoch_z * (
            self.dynamics._discrete if self.dynamics._discrete else 1
        )
        self._z_branch_feat_dim = z_stoch_dim + self.dynamics._deter_z
        self.z_action_adversary = nn.Sequential(
            nn.Linear(self._z_branch_feat_dim * 2, config.dyn_hidden),
            act_fn(),
            nn.Linear(config.dyn_hidden, config.num_actions),
        )
        self._z_action_grl_scale = getattr(config, "z_action_grl_scale", 1.0)

        # S-only Experiment 5A. This has no trainable predictor: it directly
        # regularizes the existing S prior transition used by imagination.
        sms_cfg = getattr(config, "s_multi_step_consistency", {}) or {}
        self._s_multistep_enabled = bool(sms_cfg.get("enabled", False))
        self._s_multistep_scale = float(sms_cfg.get("loss_scale", 0.0))
        self._s_multistep = None
        if self._s_multistep_enabled:
            s_stoch_dim = self.dynamics._stoch_s * (
                self.dynamics._discrete if self.dynamics._discrete else 1
            )
            s_feat_dim = s_stoch_dim + self.dynamics._deter_s
            self._s_multistep = s_multistep_consistency.SOnlyMultiStepRSSMConsistency(
                feat_dim=s_feat_dim,
                horizons=sms_cfg.get("horizons", [1, 2, 4, 8, 15]),
                horizon_weights=sms_cfg.get(
                    "horizon_weights", [1.0, 1.0, 0.75, 0.5, 0.25]
                ),
                projection_dim=int(sms_cfg.get("projection_dim", 512)),
                starts_per_sequence=int(sms_cfg.get("starts_per_sequence", 4)),
                projection_seed=int(sms_cfg.get("projection_seed", 314159)),
            )
       
        for name in config.grad_heads:
            assert name in self.heads, name

        self._model_opt = tools.Optimizer(
            "model", self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt, use_amp=self._use_amp,
        )

        print(f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables.")

        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            end=config.end_head["loss_scale"],
            intrinsic=config.intrinsic_head["loss_scale"],
            inverse=getattr(config, "inverse_loss_scale", 1.0),
            affordance_s=getattr(config, "affordance_s_scale", 1.0),
            z_action_adv=getattr(config, "z_action_adv_scale", 0.0),
        )

    def _train(self, data_origin):
        # 仅预处理基础序列数据
        data = self.preprocess(data_origin)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # 编码图像特征
                embed = self.encoder(data)

                # --- 正常序列观察 (Observe Phase) ---
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                # --- [改进1] 正确时间对齐的S分支逆动力学分类损失 ---
                # Replay中的action[t]是到达observation[t]的动作，因此
                # (state[t-1], state[t])必须监督action[t]，即data[:, 1:]。
                h_s_t = post["deter_s"][:, :-1]
                h_s_next = post["deter_s"][:, 1:]
                feat_inv_s = torch.cat([h_s_t, h_s_next], dim=-1)
                pred_action_s = self.dynamics._inverse_dynamics(feat_inv_s)

                true_action_onehot = data["action"][:, 1:]
                true_action = torch.argmax(true_action_onehot, dim=-1)
                valid_transition = 1.0 - data["is_first"][:, 1:].float()
                valid_den = torch.clamp(valid_transition.sum(), min=1.0)

                inv_per_step = F.cross_entropy(
                    pred_action_s.reshape(-1, pred_action_s.shape[-1]),
                    true_action.reshape(-1),
                    reduction="none",
                ).reshape_as(valid_transition)
                loss_inv = (inv_per_step * valid_transition).sum() / valid_den
                acc_action_s = (
                    (pred_action_s.argmax(dim=-1) == true_action).float()
                    * valid_transition
                ).sum() / valid_den

                # --- [改进2] 对抗抑制Z分支中的动作泄漏 ---
                z_stoch = post["stoch_z"]
                if self.dynamics._discrete:
                    z_stoch = z_stoch.reshape(
                        list(z_stoch.shape[:-2])
                        + [self.dynamics._stoch_z * self.dynamics._discrete]
                    )
                feat_z = torch.cat([z_stoch, post["deter_z"]], dim=-1)
                feat_inv_z = torch.cat([feat_z[:, :-1], feat_z[:, 1:]], dim=-1)
                pred_action_z = self.z_action_adversary(
                    gradient_reverse(feat_inv_z, self._z_action_grl_scale)
                )
                z_adv_per_step = F.cross_entropy(
                    pred_action_z.reshape(-1, pred_action_z.shape[-1]),
                    true_action.reshape(-1),
                    reduction="none",
                ).reshape_as(valid_transition)
                loss_z_action_adv = (z_adv_per_step * valid_transition).sum() / valid_den
                acc_action_z = (
                    (pred_action_z.argmax(dim=-1) == true_action).float()
                    * valid_transition
                ).sum() / valid_den

                # --- [改进3] Affordance仅作为S分支辅助监督，不进入encoder ---
                s_stoch_dim = self.dynamics._stoch_s * (self.dynamics._discrete if self.dynamics._discrete else 1)
                s_feat_dim = s_stoch_dim + self.dynamics._deter_s
                
                feat_all = self.dynamics.get_feat(post)
                feat_s_only = torch.zeros_like(feat_all)
                feat_s_only[..., :s_feat_dim] = feat_all[..., :s_feat_dim]
                
                preds_s = self.heads["decoder"](feat_s_only)
                loss_affordance_s = -preds_s['heatmap'].log_prob(data['heatmap'])

                # 3. 计算 KL 散度损失
                kl_free = self._config.kl_free 
                dyn_scale = self._config.dyn_scale 
                rep_scale = self._config.rep_scale 

                (
                    kl_loss_img, kl_value_img, dyn_loss_img, rep_loss_img,
                    kl_value_s, kl_value_z,
                ) = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                # 4. 计算各个预测头 (Heads) 的损失
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    
                    if isinstance(pred, dict):
                        preds.update(pred)
                    else:
                        preds[name] = pred
                        
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    losses[name] = loss
                    
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }

                # 5. S-only 5A: 从 replay posterior S 出发，用真实连续动作
                #    open-loop rollout S prior，并对齐未来 posterior S。
                s_multistep_metrics = {}
                s_multistep_loss = loss_inv * 0.0
                if self._s_multistep_enabled:
                    s_multistep_loss, s_multistep_metrics = self._s_multistep(
                        self.dynamics, post, data["action"], data["is_first"]
                    )

                # 6. 汇总总损失：保留 repaired inverse + S-Aff，关闭Z adversary，
                #    再增加 S-only 5A。
                model_loss = sum(scaled.values()) + kl_loss_img
                total_loss = (
                    torch.mean(model_loss)
                    + loss_inv * self._scales.get("inverse", 1.0)
                    + loss_z_action_adv * self._scales.get("z_action_adv", 0.0)
                    + torch.mean(loss_affordance_s)
                    * self._scales.get("affordance_s", 1.0)
                    + self._s_multistep_scale * s_multistep_loss
                )

            # 统一执行优化
            metrics = self._model_opt(total_loss, self.parameters())

        # 记录指标
        metrics.update({f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()})
        metrics["loss_inverse"] = to_np(loss_inv)
        metrics["action_acc_s"] = to_np(acc_action_s)
        metrics["loss_z_action_adv"] = to_np(loss_z_action_adv)
        metrics["action_acc_z"] = to_np(acc_action_z)
        metrics["loss_affordance_s"] = to_np(torch.mean(loss_affordance_s))
        metrics.update({name: to_np(value) for name, value in s_multistep_metrics.items()})
        metrics["s_multistep_scale"] = self._s_multistep_scale
        metrics["model_loss"] = to_np(total_loss)
        metrics["kl"] = to_np(torch.mean(kl_value_img))
        metrics["kl_s"] = to_np(torch.mean(kl_value_s))
        metrics["kl_z"] = to_np(torch.mean(kl_value_z))
        
        with torch.cuda.amp.autocast(self._use_amp):
            s_stats = {k[2:]:v for k,v in post.items() if k.startswith("s_")}
            z_stats = {k[2:]:v for k,v in post.items() if k.startswith("z_")}
            metrics["post_ent_s"] = to_np(torch.mean(self.dynamics.get_dist(s_stats).entropy()))
            metrics["post_ent_z"] = to_np(torch.mean(self.dynamics.get_dist(z_stats).entropy()))

        post = {k: v.detach() for k, v in post.items()}
        context = dict(embed=embed, feat=self.dynamics.get_feat(post), kl=kl_value_img)

        return post, None, context, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0
        obs["heatmap"] = torch.Tensor(obs["heatmap"]).unsqueeze(-1) / 255.0
        

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
            
        assert "is_first" in obs
        assert "is_terminal" in obs

        obs["end"] = torch.Tensor(obs["is_terminal"]).unsqueeze(-1)
        
        # 兼容性处理：防止 wrapper 端漏传内在奖励
        if "intrinsic" not in obs:
            obs["intrinsic"] = np.zeros_like(obs["reward"], dtype=np.float32)

        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
    
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[:6]

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            
        self.actor = networks.MLP(
            feat_size, (config.num_actions,), config.actor["layers"], config.units,
            config.act, config.norm, config.actor["dist"], config.actor["std"],
            config.actor["min_std"], config.actor["max_std"], absmax=1.0,
            temp=config.actor["temp"], unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"], name="Actor",
        )
        self.value = networks.MLP(
            feat_size, (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"], config.units, config.act, config.norm,
            config.critic["dist"], outscale=config.critic["outscale"],
            device=config.device, name="Value",
        )
        
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
            
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer("actor", self.actor.parameters(), config.actor["lr"], config.actor["eps"], config.actor["grad_clip"], **kw)
        self._value_opt = tools.Optimizer("value", self.value.parameters(), config.critic["lr"], config.critic["eps"], config.critic["grad_clip"], **kw)
        
        if self._config.reward_EMA:
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
            self,
            start,
            objective,
            intrinsic_objective,
            is_end,
        ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # 展平初始状态
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                start = {k: flatten(v) for k, v in start.items()} 

                # 标准 15 步连续想象 (替换掉原先极其复杂的跳跃 for 循环)
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )

                # 奖励计算与策略更新
                reward = objective(imag_feat, imag_state, imag_action)
                intrinsic_reward = intrinsic_objective(imag_feat, imag_state, imag_action)
                intrinsic_scale = getattr(self._config, "intrinsic_reward_scale", 1.0)
                reward = reward + intrinsic_scale * intrinsic_reward

                actor_ent = self.actor(imag_feat).entropy() 

                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, is_end
                )

                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, base
                )

                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        # Value 网络更新
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(value_input[:-1].detach())
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            action = policy(feat.detach()).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(step, [torch.arange(horizon)], (start, None, None))
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if horizon == 1:
            return feats.squeeze(0), {k: v.squeeze(0) for k, v in succ.items()}, actions.squeeze(0)
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward, is_end):
        end = is_end(imag_state) 
        gamma = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        discount = gamma * (1.0 - end)
        
        # 恢复使用最标准的 lambda_return
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()

        return target, weights, value[:-1]
    
    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics
    
    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1