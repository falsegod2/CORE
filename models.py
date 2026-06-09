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

logger = logging.getLogger(__name__)
to_np = lambda x: x.detach().cpu().numpy()

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

                # --- [创新点 1：解耦约束] 计算逆动力学损失 ---
                h_s_t = post["deter_s"][:, :-1]
                h_s_next = post["deter_s"][:, 1:]
                feat_inv = torch.cat([h_s_t, h_s_next], dim=-1)
                pred_action = self.dynamics._inverse_dynamics(feat_inv)
                
                true_action = data["action"][:, :-1]
                loss_inv = F.mse_loss(pred_action, true_action)

                # --- [创新点 2：可供性先验指导的解耦增强] ---
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

                kl_loss_img, kl_value_img, dyn_loss_img, rep_loss_img = self.dynamics.kl_loss(
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

                # 5. 汇总总损失
                model_loss = sum(scaled.values()) + kl_loss_img
                total_loss = torch.mean(model_loss) + \
                            loss_inv * self._scales.get("inverse", 1.0) + \
                            torch.mean(loss_affordance_s) * self._scales.get("affordance_s", 1.0)

            # 统一执行优化
            metrics = self._model_opt(total_loss, self.parameters())

        # 记录指标
        metrics.update({f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()})
        metrics["loss_inverse"] = to_np(loss_inv)
        metrics["loss_affordance_s"] = to_np(torch.mean(loss_affordance_s))
        metrics["model_loss"] = to_np(total_loss)
        metrics["kl"] = to_np(torch.mean(kl_value_img))
        
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
            stoch_size = (config.dyn_stoch // 2) * config.dyn_discrete
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            stoch_size = config.dyn_stoch // 2
            feat_size = config.dyn_stoch + config.dyn_deter
            
        # 2. Manager 现在只输出 stoch_size 的 \Delta S
        self.manager_actor = networks.MLP(
            feat_size, (stoch_size,), config.manager["layers"], config.units,
            config.act, config.norm, config.manager["dist"], config.manager["std"],
            config.manager["min_std"], config.manager["max_std"], absmax=1.0,
            temp=config.manager["temp"], unimix_ratio=config.manager["unimix_ratio"],
            outscale=config.manager["outscale"], name="ManagerActor",
        )
        self.manager_value = networks.MLP(
            feat_size, (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"], config.units, config.act, config.norm,
            config.critic["dist"], outscale=config.critic["outscale"],
            device=config.device, name="ManagerValue",
        )


        # ==========================================
        # 3. Worker 网络 (底层，也就是原本的 Actor，负责执行)
        # ==========================================
        # Worker Actor: 【注意这里】输入变成了 feat_size * 2 (当前状态拼接 Manager给的S_goal)
        self.actor = networks.MLP(
            feat_size + stoch_size, (config.num_actions,), config.actor["layers"], config.units,
            config.act, config.norm, config.actor["dist"], config.actor["std"],
            config.actor["min_std"], config.actor["max_std"], absmax=1.0,
            temp=config.actor["temp"], unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"], name="WorkerActor",
        )
        self.value = networks.MLP(
            feat_size + stoch_size, (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"], config.units, config.act, config.norm,
            config.critic["dist"], outscale=config.critic["outscale"],
            device=config.device, name="WorkerValue",
        )
        
        # ==========================================
        # 4. Slow Target 机制及优化器 (需要分发给 Manager 和 Worker)
        # ==========================================
        if config.critic["slow_target"]:
            self._manager_slow_value = copy.deepcopy(self.manager_value)
            self._worker_slow_value = copy.deepcopy(self.value)
            self._updates = 0
            
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)


        # 分离优化器，防止梯度互串
        self._manager_actor_opt = tools.Optimizer("manager_actor", self.manager_actor.parameters(), config.manager["lr"], config.manager["eps"], config.manager["grad_clip"], **kw)
        self._manager_value_opt = tools.Optimizer("manager_value", self.manager_value.parameters(), config.critic["lr"], config.critic["eps"], config.critic["grad_clip"], **kw)
        
        self._actor_opt = tools.Optimizer("worker_actor", self.actor.parameters(), config.actor["lr"], config.actor["eps"], config.actor["grad_clip"], **kw)
        self._value_opt = tools.Optimizer("worker_value", self.value.parameters(), config.critic["lr"], config.critic["eps"], config.critic["grad_clip"], **kw)
        
        if self._config.reward_EMA:
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(self, start, objective, intrinsic_objective, is_end):
        self._update_slow_target()
        metrics = {}

        # 1. 展开初始状态
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()} 

        # 2. 双层想象 (不再需要传 actor 进去)
        imag_feat, imag_state, imag_next_state, imag_action, imag_goal, imag_manager_action, imag_manager_mask = self._imagine(
            start, self._config.imag_horizon
        )

        # ==========================================
        # 3. Manager (高层) 优化
        # ==========================================
        with tools.RequiresGrad(self.manager_actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Manager 关注环境真实奖励 + MineCLIP 语义奖励
                reward = objective(imag_feat, imag_state, imag_action)
                intrinsic_reward = intrinsic_objective(imag_feat, imag_state, imag_action)
                intrinsic_scale = getattr(self._config, "intrinsic_reward_scale", 1.0)
                m_reward = reward + intrinsic_scale * intrinsic_reward

                m_target, m_weights, m_base = self._compute_target(
                    imag_feat, imag_state, m_reward, is_end, self.manager_value
                )

                # 计算 Manager Advantage 并严格应用 EMA (极其重要)
                m_target_st = torch.stack(m_target, dim=1)
                if self._config.reward_EMA:
                    offset, scale = self.reward_ema(m_target_st, self.ema_vals)
                    m_normed_target = (m_target_st - offset) / scale
                    m_normed_base = (m_base - offset) / scale
                    m_adv = m_normed_target - m_normed_base
                    metrics["Manager_EMA_005"] = to_np(self.ema_vals[0])
                    metrics["Manager_EMA_095"] = to_np(self.ema_vals[1])
                else:
                    m_adv = m_target_st - m_base

                m_policy = self.manager_actor(imag_feat.detach())
                m_log_prob = m_policy.log_prob(imag_manager_action)[:-1][:, :, None]

                # 【修复】极其鲁棒的动态扩维，强制与 m_weights 的维度对齐
                imag_manager_mask = imag_manager_mask.to(device=m_weights.device, dtype=m_weights.dtype)
                while imag_manager_mask.dim() < m_weights.dim():
                    imag_manager_mask = imag_manager_mask.unsqueeze(-1)

                m_actor_loss = -m_weights[:-1] * m_log_prob * m_adv.detach() * imag_manager_mask[:-1]
                
                m_ent = m_policy.entropy()
                m_actor_loss -= self._config.actor["entropy"] * m_ent[:-1, ..., None] * imag_manager_mask[:-1]
                m_actor_loss = torch.mean(m_actor_loss)

        # ==========================================
        # 4. Worker (底层) 优化
        # ==========================================
        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Worker 奖励必须评价“动作后是否更接近 Manager goal”。
                # current_stoch 对应执行 action 前的 s_t；next_stoch 对应 dynamics.img_step 后的 s_{t+1}。
                current_stoch = imag_state["stoch_s"]
                next_stoch = imag_next_state["stoch_s"]
                if len(current_stoch.shape) > 3:
                    current_stoch = current_stoch.reshape(current_stoch.shape[0], current_stoch.shape[1], -1)
                if len(next_stoch.shape) > 3:
                    next_stoch = next_stoch.reshape(next_stoch.shape[0], next_stoch.shape[1], -1)

                w_reward, w_progress, w_cos_before, w_cos_after = self._compute_worker_reward(
                    current_stoch, next_stoch, imag_goal
                )

                w_inp = torch.cat([imag_feat, imag_goal], dim=-1)
                w_target, w_weights, w_base = self._compute_target(
                    w_inp, imag_state, w_reward, is_end, self.value
                )

                # Worker 奖励已经被严格缩放到 [0,1]，无需 EMA 干扰
                w_target_st = torch.stack(w_target, dim=1)
                w_adv = w_target_st - w_base

                w_policy = self.actor(w_inp.detach())
                w_log_prob = w_policy.log_prob(imag_action)[:-1][:, :, None]
                w_actor_loss = -w_weights[:-1] * w_log_prob * w_adv.detach()

                w_ent = w_policy.entropy()
                w_actor_loss -= self._config.actor["entropy"] * w_ent[:-1, ..., None]
                w_actor_loss = torch.mean(w_actor_loss)

        # ==========================================
        # 5. Value Networks (Critic) 优化
        # ==========================================
        with tools.RequiresGrad(self.manager_value):
            with torch.cuda.amp.autocast(self._use_amp):
                m_val = self.manager_value(imag_feat[:-1].detach())
                m_val_loss = -m_val.log_prob(m_target_st.detach())
                if self._config.critic["slow_target"]:
                    m_slow_val = self._manager_slow_value(imag_feat[:-1].detach())
                    m_val_loss -= m_val.log_prob(m_slow_val.mode().detach())
                m_val_loss = torch.mean(m_weights[:-1] * m_val_loss[:, :, None])

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                w_val = self.value(w_inp[:-1].detach())
                w_val_loss = -w_val.log_prob(w_target_st.detach())
                if self._config.critic["slow_target"]:
                    w_slow_val = self._worker_slow_value(w_inp[:-1].detach())
                    w_val_loss -= w_val.log_prob(w_slow_val.mode().detach())
                w_val_loss = torch.mean(w_weights[:-1] * w_val_loss[:, :, None])

        # ==========================================
        # 6. 梯度截断与统一更新
        # ==========================================
        with tools.RequiresGrad(self):
            metrics.update(self._manager_actor_opt(m_actor_loss, self.manager_actor.parameters()))
            metrics.update(self._manager_value_opt(m_val_loss, self.manager_value.parameters()))
            metrics.update(self._actor_opt(w_actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(w_val_loss, self.value.parameters()))

        # 记录关键指标以便于您在 TensorBoard 观察
        metrics.update(tools.tensorstats(m_val.mode(), "manager_value"))
        metrics.update(tools.tensorstats(w_val.mode(), "worker_value"))
        metrics.update(tools.tensorstats(m_reward, "manager_reward_total"))
        # 【闭环修复 3】: 单独记录环境和 CLIP 奖励，证明 Manager 的目标来源
        metrics.update(tools.tensorstats(reward, "manager_reward_env"))
        metrics.update(tools.tensorstats(intrinsic_reward, "manager_reward_clip"))
        
        metrics.update(tools.tensorstats(w_reward, "worker_reward"))
        metrics.update(tools.tensorstats(w_progress, "worker_progress"))
        metrics.update(tools.tensorstats(w_cos_before, "worker_cos_before"))
        metrics.update(tools.tensorstats(w_cos_after, "worker_cos_after"))
        metrics.update(tools.tensorstats(m_ent, "manager_entropy"))
        metrics.update(tools.tensorstats(w_ent, "worker_entropy"))

        with torch.no_grad():
            action_idx = torch.argmax(imag_action, dim=-1)
            for i in range(self._config.num_actions):
                metrics[f"worker_action_{i}_frac"] = to_np((action_idx == i).float().mean())

        # 记录 Manager 跳跃幅度和更新比例
        metrics["manager_update_ratio"] = to_np(imag_manager_mask.mean())
        metrics["manager_jump_norm"] = to_np(torch.norm(imag_manager_action, p=2, dim=-1).mean())
        metrics["manager_goal_norm"] = to_np(torch.norm(imag_goal, p=2, dim=-1).mean())
        
        return imag_feat, imag_state, imag_action, w_weights, metrics

    def _compute_worker_reward(self, current_feat, next_feat, goal_feat):
        import torch.nn.functional as F
        cos_before = F.cosine_similarity(current_feat, goal_feat, dim=-1)
        cos_after = F.cosine_similarity(next_feat, goal_feat, dim=-1)
        progress = cos_after - cos_before

        scale = getattr(self._config, "worker_reward_scale", 5.0)
        clip = getattr(self._config, "worker_reward_clip", 1.0)
        worker_reward = torch.clamp(scale * progress, -clip, clip)

        return (
            worker_reward.unsqueeze(-1),
            progress.unsqueeze(-1),
            cos_before.unsqueeze(-1),
            cos_after.unsqueeze(-1),
        )

    def _imagine(self, start, horizon):
        dynamics = self._world_model.dynamics

        def step(prev, t):
            # 【修复1】这里必须是 6 个变量来接收，多加一个 '_'
            state, prev_goal, _, _, _, _ = prev
            feat = dynamics.get_feat(state)
            
            # 【提取当前的物理 受控 stoch 内容】
            stoch_feat = state["stoch_s"]
            if len(stoch_feat.shape) > 2:
                stoch_feat = stoch_feat.reshape(stoch_feat.shape[0], -1)

            if prev_goal is None:
                prev_goal = torch.zeros_like(stoch_feat)

            update_mask = (t % self._config.manager_freq == 0).to(device=stoch_feat.device, dtype=stoch_feat.dtype)

            manager_dist = self.manager_actor(feat.detach())
            manager_action = manager_dist.sample()
            
            import torch.nn.functional as F
            manager_direction = F.normalize(manager_action, p=2, dim=-1)
            goal_scale = getattr(self._config, "goal_scale", 1.0)
            scaled_action = manager_direction * goal_scale
            
            new_goal = (stoch_feat.detach() + scaled_action).detach()
            goal_stoch = update_mask * new_goal + (1.0 - update_mask) * prev_goal

            worker_inp = torch.cat([feat, goal_stoch], dim=-1)
            action = self.actor(worker_inp.detach()).sample()

            succ = dynamics.img_step(state, action)
            
            # 【修复2】这里必须 return 6 个返回值，加上 update_mask
            return succ, goal_stoch, feat, action, manager_action, update_mask

        # 调用原生的 static_scan
        succ, goals, feats, actions, manager_actions, manager_masks = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None, None, None)
        )
        
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        next_states = succ
        
        if horizon == 1:
            return (
                feats.squeeze(0),
                {k: v.squeeze(0) for k, v in states.items()},
                {k: v.squeeze(0) for k, v in next_states.items()},
                actions.squeeze(0),
                goals.squeeze(0),
                manager_actions.squeeze(0),
                manager_masks.squeeze(0),
            )
            
        return feats, states, next_states, actions, goals, manager_actions, manager_masks

    def _compute_target(self, value_input, imag_state, reward, is_end, value_net):
        end = is_end(imag_state) 
        gamma = self._config.discount * torch.ones_like(reward)
        # 显式使用传入的 value_net
        value = value_net(value_input).mode()
        discount = gamma * (1.0 - end)
        
        # 原汁原味的 lambda_return
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
                # 1. 更新 Worker 的 Slow Target
                for s, d in zip(self.value.parameters(), self._worker_slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
                # 2. 更新 Manager 的 Slow Target
                for s, d in zip(self.manager_value.parameters(), self._manager_slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1