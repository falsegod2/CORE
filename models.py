# coding=utf-8
import copy
import logging

import numpy as np
import torch
import torch.nn as nn

import networks
import tools

logger = logging.getLogger(__name__)
to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """DreamerV3 percentile-based return normalization."""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(flat_x, self.range)
        ema_vals[:] = self.alpha * x_quantile + (1.0 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    """Single-RSSM DreamerV3 world model with optional MineCLIP shaping head."""

    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        del act_space, step
        self._use_amp = config.precision == 16
        self._config = config

        shapes = {key: tuple(space.shape) for key, space in obs_space.spaces.items()}
        object_config = config.task_object_tokens
        if object_config.get("enabled", False):
            self.encoder = networks.TaskRelevantObjectEncoder(
                shapes, config.encoder, object_config
            )
        else:
            self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )

        feat_size = (
            config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            if config.dyn_discrete
            else config.dyn_stoch + config.dyn_deter
        )

        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["end"] = networks.MLP(
            feat_size,
            (),
            config.end_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.end_head["outscale"],
            device=config.device,
            name="End",
        )
        self.heads["mineclip_reward"] = networks.MLP(
            feat_size,
            (255,) if config.mineclip_head["dist"] == "symlog_disc" else (),
            config.mineclip_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.mineclip_head["dist"],
            outscale=config.mineclip_head["outscale"],
            device=config.device,
            name="MineCLIPReward",
        )

        for name in config.grad_heads:
            if name not in self.heads:
                raise KeyError(f"Unknown gradient head: {name}")

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            "Optimizer model_opt has "
            f"{sum(parameter.numel() for parameter in self.parameters())} variables."
        )

        self._scales = {
            "reward": config.reward_head["loss_scale"],
            "end": config.end_head["loss_scale"],
            "mineclip_reward": config.mineclip_head["loss_scale"],
        }

    def _train(self, data_origin):
        data = self.preprocess(data_origin)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                if kl_loss.shape != embed.shape[:2]:
                    raise RuntimeError(
                        f"KL shape {kl_loss.shape} does not match sequence shape "
                        f"{embed.shape[:2]}"
                    )

                feat = self.dynamics.get_feat(post)
                preds = {}
                for name, head in self.heads.items():
                    head_input = feat if name in self._config.grad_heads else feat.detach()
                    pred = head(head_input)
                    if isinstance(pred, dict):
                        preds.update(pred)
                    else:
                        preds[name] = pred

                losses = {}
                for name, pred in preds.items():
                    if name not in data:
                        raise KeyError(
                            f"World-model head '{name}' has no target in replay data"
                        )
                    loss = -pred.log_prob(data[name])
                    if loss.shape != embed.shape[:2]:
                        raise RuntimeError(
                            f"Loss shape for {name}: {loss.shape}; expected "
                            f"{embed.shape[:2]}"
                        )
                    losses[name] = loss

                scaled_losses = {
                    name: loss * self._scales.get(name, 1.0)
                    for name, loss in losses.items()
                }
                encoder_aux_losses = {}
                if hasattr(self.encoder, "get_aux_losses"):
                    encoder_aux_losses = self.encoder.get_aux_losses()
                    for name, loss in encoder_aux_losses.items():
                        if loss.shape != embed.shape[:2]:
                            raise RuntimeError(
                                f"Encoder auxiliary loss {name} has shape "
                                f"{loss.shape}; expected {embed.shape[:2]}"
                            )
                model_loss = (
                    sum(scaled_losses.values())
                    + sum(encoder_aux_losses.values())
                    + kl_loss
                )
                mean_model_loss = torch.mean(model_loss)

            metrics = self._model_opt(mean_model_loss, self.parameters())

        metrics.update(
            {
                f"{name}_loss": to_np(torch.mean(loss))
                for name, loss in losses.items()
            }
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["model_loss"] = to_np(mean_model_loss)
        for name, loss in encoder_aux_losses.items():
            metrics[f"{name}_scaled_loss"] = to_np(torch.mean(loss))
        if hasattr(self.encoder, "get_metrics"):
            metrics.update({
                key: to_np(value) if torch.is_tensor(value) else value
                for key, value in self.encoder.get_metrics().items()
            })

        with torch.cuda.amp.autocast(self._use_amp):
            prior_ent = self.dynamics.get_dist(prior).entropy()
            post_ent = self.dynamics.get_dist(post).entropy()
            metrics["prior_ent"] = to_np(torch.mean(prior_ent))
            metrics["post_ent"] = to_np(torch.mean(post_ent))
            context = {
                "embed": embed,
                "feat": self.dynamics.get_feat(post),
                "kl": kl_value,
                "postent": post_ent,
            }

        post = {key: value.detach() for key, value in post.items()}
        return post, None, context, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.as_tensor(obs["image"], dtype=torch.float32) / 255.0

        if "discount" in obs:
            obs["discount"] = torch.as_tensor(
                obs["discount"] * self._config.discount,
                dtype=torch.float32,
            ).unsqueeze(-1)

        if "is_first" not in obs or "is_terminal" not in obs:
            raise KeyError("Replay observations require is_first and is_terminal")

        obs["end"] = torch.as_tensor(
            obs["is_terminal"], dtype=torch.float32
        ).unsqueeze(-1)

        if "mineclip_reward" not in obs:
            obs["mineclip_reward"] = np.zeros_like(
                obs["reward"], dtype=np.float32
            )

        object_config = self._config.task_object_tokens
        task_key = object_config.get("task_key", "task_embedding")
        visual_key = object_config.get(
            "visual_key", "mineclip_embedding"
        )
        prefix_shape = np.asarray(obs["image"]).shape[:-3]
        if task_key not in obs:
            task_dim = int(object_config.get("task_dim", 512))
            obs[task_key] = np.zeros(
                prefix_shape + (task_dim,), dtype=np.float32
            )
        if visual_key not in obs:
            visual_dim = int(object_config.get("visual_dim", 512))
            obs[visual_key] = np.zeros(
                prefix_shape + (visual_dim,), dtype=np.float32
            )

        return {
            key: value.to(self._config.device)
            if torch.is_tensor(value)
            else torch.as_tensor(
                value, dtype=torch.float32, device=self._config.device
            )
            for key, value in obs.items()
        }

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](
            self.dynamics.get_feat(states)
        )["image"].mode()[:6]

        init = {key: value[:, -1] for key, value in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        open_loop = self.heads["decoder"](
            self.dynamics.get_feat(prior)
        )["image"].mode()

        model = torch.cat([recon[:, :5], open_loop], 1)
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
            mineclip_objective,
            is_end,
        ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Flatten posterior states from replay into imagination starts.
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                start = {k: flatten(v) for k, v in start.items()} 

                # Standard one-step RSSM rollout repeated for the configured horizon
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )

                # Predict rewards along imagined trajectories.
                reward = objective(imag_feat, imag_state, imag_action)
                mineclip_reward = mineclip_objective(imag_feat, imag_state, imag_action)
                reward = reward + self._config.mineclip_reward_scale * mineclip_reward

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

        # Critic update.
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
        
        # Standard lambda-return
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