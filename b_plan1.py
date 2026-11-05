"""B-Plan1's RGB world model and replay-trained policy for expr.py."""

import numpy as np
import torch
from torch import nn

import networks
import tools
from b_value_planner import BConfig, RealValuePlanner


class BWorldModel(nn.Module):
    def __init__(self, obs_space, config):
        super().__init__()
        self._config = config
        self._use_amp = config.precision == 16
        shapes = {"image": tuple(obs_space.spaces["image"].shape)}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
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
            self.encoder.outdim,
            config.device,
            extra_jump_action=False,
        )
        self.feat_size = config.dyn_deter + config.dyn_stoch * (config.dyn_discrete or 1)
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(self.feat_size, shapes, **config.decoder)
        self.heads["reward"] = self._scalar_head(config.reward_head, "Reward")
        for channel in ("intrinsic_mineclip", "intrinsic_heatmap"):
            self.heads[channel] = self._scalar_head(config.intrinsic_head, "TeacherBonus")
        self.heads["end"] = networks.MLP(
            self.feat_size, (), config.end_head["layers"], config.units, config.act,
            config.norm, dist="binary", outscale=config.end_head["outscale"],
            device=config.device, name="End",
        )
        for name in config.grad_heads:
            assert name in self.heads, name

        self._scales = {
            "reward": config.reward_head["loss_scale"],
            "end": config.end_head["loss_scale"],
            "intrinsic_mineclip": (
                config.intrinsic_head["loss_scale"] if config.b_mineclip_intrinsic else 0.0
            ),
            "intrinsic_heatmap": 0.0,
        }
        self._model_opt = tools.Optimizer(
            "model", self.parameters(), config.model_lr, config.opt_eps,
            config.grad_clip, config.weight_decay, opt=config.opt, use_amp=self._use_amp,
        )
        print(f"Optimizer model_opt has {sum(p.numel() for p in self.parameters())} variables.")

    def _scalar_head(self, spec, name):
        config = self._config
        return networks.MLP(
            self.feat_size, (255,) if spec["dist"] == "symlog_disc" else (),
            spec["layers"], config.units, config.act, config.norm, dist=spec["dist"],
            outscale=spec["outscale"], device=config.device, name=name,
        )

    def preprocess(self, obs):
        # The shared simulator also returns LS-only fields, some of which
        # contain None when there is no zoomed observation.
        keys = (
            "image", "is_first", "is_terminal", "action", "reward",
            "intrinsic_mineclip", "intrinsic_heatmap",
        )
        data = {
            key: torch.as_tensor(obs[key], dtype=torch.float32, device=self._config.device)
            for key in keys if key in obs
        }
        data["image"] = data["image"] / 255.0
        data["end"] = data["is_terminal"].unsqueeze(-1)
        return data

    def _train(self, replay):
        data = self.preprocess(replay)
        config = self._config
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, config.kl_free, config.dyn_scale, config.rep_scale,
                )
                feat = self.dynamics.get_feat(post)
                losses = {}
                for name, head in self.heads.items():
                    head_input = feat if name in config.grad_heads else feat.detach()
                    pred = head(head_input)
                    predictions = pred if isinstance(pred, dict) else {name: pred}
                    for key, dist in predictions.items():
                        loss = -dist.log_prob(data[key])
                        assert loss.shape == embed.shape[:2], (key, loss.shape)
                        losses[key] = loss
                model_loss = kl_loss + sum(
                    loss * self._scales.get(key, 1.0) for key, loss in losses.items()
                )
            metrics = self._model_opt(model_loss.mean(), self.parameters())

        metrics.update({f"{key}_loss": loss.detach().mean().item() for key, loss in losses.items()})
        metrics.update(
            kl_free=config.kl_free,
            dyn_scale=config.dyn_scale,
            rep_scale=config.rep_scale,
            dyn_loss=dyn_loss.detach().mean().item(),
            rep_loss=rep_loss.detach().mean().item(),
            kl=kl_value.detach().mean().item(),
            model_loss=model_loss.detach().mean().item(),
            prior_ent=self.dynamics.get_dist(prior).entropy().detach().mean().item(),
            post_ent=self.dynamics.get_dist(post).entropy().detach().mean().item(),
        )
        return {key: value.detach() for key, value in post.items()}, metrics

    @torch.no_grad()
    def video_pred(self, replay):
        data = self.preprocess(replay)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5],
        )
        decoder = self.heads["decoder"]
        recon = decoder(self.dynamics.get_feat(states))["image"].mode()
        if embed.shape[1] > 5:
            last = {key: value[:, -1] for key, value in states.items()}
            prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], last)
            open_loop = decoder(self.dynamics.get_feat(prior))["image"].mode()
            recon = torch.cat([recon, open_loop], dim=1)
        truth = data["image"][:6]
        return torch.cat([truth, recon, (recon - truth + 1.0) / 2.0], dim=2)


class RealAnchoredAgent(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super().__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_train = tools.Every(config.batch_size * config.batch_length / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = BWorldModel(obs_space, config)
        self._value = RealValuePlanner(BConfig(
            actions=config.num_actions,
            feat_dim=self._wm.feat_size,
            discount=config.discount,
            plan_start=config.b_plan_start,
            mineclip_intrinsic_enabled=config.b_mineclip_intrinsic,
        ))
        print(f"[B-Plan1] One-step planning starts at {config.b_plan_start} training steps. "
              f"MineCLIP intrinsic: {config.b_mineclip_intrinsic}.")

    def __call__(self, obs, reset, state=None, training=True):
        if training:
            steps = self._config.pretrain if self._should_pretrain() else self._should_train(self._step)
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics.setdefault("update_count", []).append(self._update_count)
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                self._metrics.clear()
                if self._config.video_pred_log:
                    open_loop = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", open_loop.cpu().numpy())
                self._logger.write(fps=True)
        output, state = self._policy(obs, state, training)
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return output, state

    @torch.no_grad()
    def _policy(self, obs, state, training):
        latent, previous_action = (None, None) if state is None else state
        data = self._wm.preprocess(obs)
        embed = self._wm.encoder(data)
        latent, _ = self._wm.dynamics.obs_step(latent, previous_action, embed, data["is_first"])
        if not training and self._config.eval_state_mean:
            latent["stoch"] = self._wm.dynamics.get_dist(latent).mode()
        action, logprob, diagnostics = self._value.choose(latent, self._wm, self._step, training)
        if training:
            for key, value in diagnostics.items():
                self._metrics.setdefault(key, []).append(value)
        return {"action": action, "logprob": logprob}, (latent, action)

    def _train(self, data):
        # B disables ConcentrationWrapper in make_env. When ClipWrapper is
        # enabled, the shared intrinsic field therefore contains only MineCLIP.
        intrinsic = data["intrinsic"] if self._config.b_mineclip_intrinsic else np.zeros_like(data["reward"])
        replay = {**data, "intrinsic_mineclip": intrinsic, "intrinsic_heatmap": np.zeros_like(intrinsic)}
        post, model_metrics = self._wm._train(replay)
        value_metrics = self._value.train_real(post, replay, self._wm)
        for key, value in {**model_metrics, **value_metrics}.items():
            self._metrics.setdefault(key, []).append(float(value))
