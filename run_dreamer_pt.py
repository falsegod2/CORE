"""Additive Dreamer-PT entry point for LS-Imagine_with_B-Plan1.

Uses the earlier PyTorch imagined actor-critic with the shared RGB RSSM.
This is not the official JAX DreamerV3 implementation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
from torch import nn
from torch.nn import functional as F

import tools
from b_plan1 import BWorldModel


ROOT = Path(__file__).resolve().parent


class ImaginedActorCritic(nn.Module):
    """Train a discrete policy and slow-target critic on RSSM rollouts."""

    def __init__(self, feat_dim, actions, config):
        super().__init__()
        self.config = config
        hidden = config.dreamer_hidden

        def make_head(out_dim):
            return nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )

        self.actor = make_head(actions).to(config.device)
        self.value = make_head(1).to(config.device)
        self.value_target = make_head(1).to(config.device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_target.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.dreamer_actor_lr)
        self.critic_opt = torch.optim.Adam(self.value.parameters(), lr=config.dreamer_critic_lr)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def dist(self, feat):
        return tools.OneHotDist(logits=self.actor(feat.float()), unimix_ratio=0.01)

    def train_imagined(self, post, wm):
        self.actor.requires_grad_(True)
        self.value.requires_grad_(True)
        config = self.config
        states = {k: v.detach().reshape(-1, *v.shape[2:]) for k, v in post.items()}
        features, rewards, discounts, bonuses = [], [], [], []
        logprobs, entropies = [], []
        feat = wm.dynamics.get_feat(states).float()
        for _ in range(config.dreamer_horizon):
            policy = self.dist(feat)
            action = policy.sample().detach().float()
            logprobs.append(policy.log_prob(action))
            entropies.append(policy.entropy())
            features.append(feat)
            with torch.no_grad():
                states = wm.dynamics.img_step(states, action, sample=True)
                feat = wm.dynamics.get_feat(states).float()
                reward = wm.heads["reward"](feat).mode().float().reshape(-1)
                bonus = (
                    wm.heads["intrinsic_mineclip"](feat).mode().float().reshape(-1)
                    if config.b_mineclip_intrinsic
                    else torch.zeros_like(reward)
                )
                terminal = wm.heads["end"](feat).mean.float().reshape(-1).clamp(0, 1)
                rewards.append(reward + bonus)
                discounts.append(config.discount * (1.0 - terminal))
                bonuses.append(bonus)

        with torch.no_grad():
            # The reward on each transition is predicted from its next state.
            tails = [self.value_target(x).squeeze(-1) for x in (*features[1:], feat)]
            target = tails[-1]
            returns = []
            for reward, discount, tail in zip(reversed(rewards), reversed(discounts), reversed(tails)):
                target = reward + discount * ((1 - config.discount_lambda) * tail + config.discount_lambda * target)
                returns.append(target)
            returns = torch.stack(returns[::-1])

        values = torch.stack([self.value(x).squeeze(-1) for x in features])
        advantage = returns - values.detach()
        advantage = advantage / advantage.abs().mean().clamp_min(1.0)
        entropy = torch.stack(entropies).mean()
        actor_loss = -(torch.stack(logprobs) * advantage).mean() - 3e-4 * entropy
        value_loss = F.smooth_l1_loss(values, returns)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        value_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), 100.0)
        self.critic_opt.step()

        with torch.no_grad():
            self.updates.add_(1)
            for slow, online in zip(self.value_target.parameters(), self.value.parameters()):
                slow.mul_(0.98).add_(online, alpha=0.02)
        return {
            "dreamer_actor_loss": actor_loss.item(),
            "dreamer_value_loss": value_loss.item(),
            "dreamer_imag_reward_mean": torch.stack(rewards).mean().item(),
            "dreamer_imag_mineclip_mean": torch.stack(bonuses).mean().item(),
            "dreamer_imag_heatmap_mean": 0.0,
            "dreamer_actor_entropy": entropy.item(),
            "dreamer_actor_grad_norm": actor_norm.item(),
            "dreamer_value_grad_norm": value_norm.item(),
        }


class DreamerPTAgent(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super().__init__()
        self._config = config
        self._logger = logger
        self._dataset = dataset
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._wm = BWorldModel(obs_space, config)
        self._behavior = ImaginedActorCritic(self._wm.feat_size, config.num_actions, config)
        print(f"[Dreamer-PT] RGB RSSM + imagined actor-critic, MineCLIP={config.b_mineclip_intrinsic}.")

    def __call__(self, obs, reset, state=None, training=True):
        if training:
            steps = self._config.pretrain if self._should_pretrain() else self._should_train(self._step)
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                self._logger.scalar("update_count", self._update_count)
                self._metrics.clear()
                self._logger.write(fps=True)

        output, state = self._policy(obs, state, training)
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return output, state

    @torch.no_grad()
    def _policy(self, obs, state, training):
        latent, last_action = (None, None) if state is None else state
        data = self._wm.preprocess(obs)
        embed = self._wm.encoder(data)
        dynamics = self._wm.dynamics
        latent, _ = dynamics.obs_step(latent, last_action, embed, data["is_first"])
        if not training and self._config.eval_state_mean:
            latent["stoch"] = latent["mean"] if "mean" in latent else dynamics.get_dist(latent).mode()
        policy = self._behavior.dist(dynamics.get_feat(latent))
        action = (
            policy.sample().float() if training
            else F.one_hot(policy.logits.argmax(-1), self._config.num_actions).float()
        )
        return {"action": action, "logprob": policy.log_prob(action)}, (latent, action)

    def _train(self, data):
        intrinsic = data["intrinsic"] if self._config.b_mineclip_intrinsic else np.zeros_like(data["reward"])
        replay = {**data, "intrinsic_mineclip": intrinsic, "intrinsic_heatmap": np.zeros_like(intrinsic)}
        post, metrics = self._wm._train(replay)
        metrics.update(self._behavior.train_imagined(post, self._wm))
        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append(float(value))


def read_config(argv=None):
    profiles = yaml.YAML(typ="safe").load((ROOT / "configs.yaml").read_text(encoding="utf-8"))
    config = {}
    for profile in ("defaults", "minedojo", "b_plan1"):
        for key, value in profiles[profile].items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="minedojo_harvest_log_in_plains")
    parser.add_argument("--target_item", default=config["target_item"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--device", default=config["device"])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--logdir", default="./logdir_dreamer_pt")
    parser.add_argument("--mineclip-intrinsic", choices=("on", "off"), default="off")
    parser.add_argument("--dreamer_hidden", type=int, default=512)
    parser.add_argument("--dreamer_horizon", type=int, default=15)
    parser.add_argument("--dreamer_actor_lr", type=float, default=3e-5)
    parser.add_argument("--dreamer_critic_lr", type=float, default=3e-5)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved settings without starting MineDojo.")
    for key in (
        "prefill", "pretrain", "batch_size", "batch_length", "train_ratio",
        "eval_every", "eval_episode_num", "log_every",
        "dyn_deter", "dyn_hidden", "units", "dyn_stoch", "dyn_discrete",
    ):
        parser.add_argument(f"--{key}", type=int, default=config[key])
    overrides = vars(parser.parse_args(argv))
    if not overrides["task"].startswith("minedojo_"):
        parser.error("--task must start with minedojo_")
    if overrides["dreamer_hidden"] < 1 or overrides["dreamer_horizon"] < 2:
        parser.error("--dreamer_hidden must be positive and --dreamer_horizon must be at least 2")
    if overrides["dreamer_actor_lr"] <= 0 or overrides["dreamer_critic_lr"] <= 0:
        parser.error("Dreamer learning rates must be positive")
    config.update(overrides)
    config["b_mineclip_intrinsic"] = overrides["mineclip_intrinsic"] == "on"
    config["logdir"] = str(Path(config["logdir"]).expanduser().resolve())
    config["results_dir"] = str(Path(config["logdir"]) / "env_diagnostics")
    config["name"] = "dreamer_pt"
    return argparse.Namespace(**config)


def main(argv=None):
    config = read_config(argv)
    if config.dry_run:
        keys = (
            "task", "seed", "device", "steps", "logdir", "results_dir", "mineclip_intrinsic",
            "envs", "prefill", "eval_every", "eval_episode_num", "dreamer_horizon",
        )
        print(json.dumps({"method": "dreamer_pt", **{key: getattr(config, key) for key in keys}}, indent=2))
        return

    os.chdir(ROOT)
    import b_plan1
    import expr

    # expr.main uses this factory for the existing RGB-only environment path.
    # The binding is changed only in this interpreter, for this run.
    original_agent = b_plan1.RealAnchoredAgent
    b_plan1.RealAnchoredAgent = DreamerPTAgent
    try:
        expr.main(config)
    finally:
        b_plan1.RealAnchoredAgent = original_agent


if __name__ == "__main__":
    main()
