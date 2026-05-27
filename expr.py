import argparse
import functools
import os
import pathlib
import sys
import torch
import numpy as np
import ruamel.yaml as yaml
from torch import nn
from torch import distributions as torchd
from datetime import datetime

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

os.environ["MUJOCO_GL"] = "osmesa"
sys.path.append(str(pathlib.Path(__file__).parent))

to_np = lambda x: x.detach().cpu().numpy()


def _metric_to_float(value):
    """Convert logger metrics to a Python float, including CUDA tensors."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(np.nanmean(value))
    if isinstance(value, (list, tuple)):
        vals = [_metric_to_float(v) for v in value if v is not None]
        if not vals:
            return 0.0
        return float(np.nanmean(vals))
    try:
        return float(value)
    except (TypeError, ValueError):
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        return float(np.nanmean(arr))


class LS_Imagine(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(LS_Imagine, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        # CORE2-SLP: latent landmark bank mined from replay by MineCLIP score.
        self._slp_landmarks = None
        self._slp_scores = None
        self._slp_bank_ptr = 0
        self._slp_bank_count = 0
        self._slp_last_metrics = {}
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, _metric_to_float(values))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _slp_scale(self):
        if not getattr(self._config, "use_slp_reward", False):
            return 0.0
        base = float(getattr(self._config, "slp_reward_scale", 0.03))
        min_scale = float(getattr(self._config, "slp_min_scale", 0.0))
        decay_steps = int(getattr(self._config, "slp_decay_steps", 200000))
        if decay_steps <= 0:
            return base
        decay = max(0.0, 1.0 - float(self._update_count) / float(decay_steps))
        return max(min_scale, base * decay)

    def _ensure_slp_bank(self, feat_dim):
        bank_size = int(getattr(self._config, "slp_bank_size", 4096))
        device = self._config.device
        if self._slp_landmarks is None or self._slp_landmarks.shape[-1] != feat_dim:
            self._slp_landmarks = torch.zeros(bank_size, feat_dim, device=device, dtype=torch.float32)
            self._slp_scores = torch.full((bank_size,), -1.0e9, device=device, dtype=torch.float32)
            self._slp_bank_ptr = 0
            self._slp_bank_count = 0

    def _get_slp_scores_from_batch(self, data):
        keys = [getattr(self._config, "slp_score_key", "obs_reward"), "obs_reward", "score"]
        for key in keys:
            if key in data:
                score = data[key]
                if isinstance(score, torch.Tensor):
                    score = score.detach().to(self._config.device).float()
                else:
                    score = torch.as_tensor(score, device=self._config.device).float()
                return score.squeeze(-1) if score.ndim > 2 and score.shape[-1] == 1 else score
        return None

    def _insert_slp_landmarks_diverse(self, cand_feat, cand_score):
        """Insert semantically strong but diverse landmarks into the bank.

        CORE2-SLP differs from plain LGP here: it avoids filling the whole bank
        with near-duplicate high-MineCLIP-score frames, and replaces an existing
        near-duplicate only when the new candidate has a better semantic score.
        """
        if cand_feat.numel() == 0:
            return 0, 0, 0
        self._ensure_slp_bank(cand_feat.shape[-1])
        bank_size = int(self._slp_landmarks.shape[0])
        use_div = bool(getattr(self._config, "slp_use_diversity", True))
        replace_similar = bool(getattr(self._config, "slp_replace_similar", True))
        sim_threshold = float(getattr(self._config, "slp_landmark_sim_threshold", 0.95))
        inserted, replaced, skipped = 0, 0, 0

        for feat_i, score_i in zip(cand_feat, cand_score):
            score_i = score_i.detach().float()
            feat_i = feat_i.detach().float()
            if self._slp_bank_count <= 0:
                idx = int(self._slp_bank_ptr)
                self._slp_landmarks[idx] = feat_i
                self._slp_scores[idx] = score_i
                self._slp_bank_ptr = int((self._slp_bank_ptr + 1) % bank_size)
                self._slp_bank_count = min(bank_size, self._slp_bank_count + 1)
                inserted += 1
                continue

            if use_div:
                valid = self._slp_landmarks[: int(self._slp_bank_count)]
                if getattr(self._config, "slp_distance", "cosine") == "cosine":
                    sim = torch.matmul(valid, feat_i)
                else:
                    sim = -torch.sum((valid - feat_i.unsqueeze(0)) ** 2, dim=-1)
                max_sim, max_idx = torch.max(sim, dim=0)
                is_duplicate = bool(max_sim.item() >= sim_threshold) if getattr(self._config, "slp_distance", "cosine") == "cosine" else False
                if is_duplicate:
                    idx = int(max_idx.item())
                    if replace_similar and score_i > self._slp_scores[idx]:
                        self._slp_landmarks[idx] = feat_i
                        self._slp_scores[idx] = score_i
                        replaced += 1
                    else:
                        skipped += 1
                    continue

            idx = int(self._slp_bank_ptr)
            self._slp_landmarks[idx] = feat_i
            self._slp_scores[idx] = score_i
            self._slp_bank_ptr = int((self._slp_bank_ptr + 1) % bank_size)
            self._slp_bank_count = min(bank_size, self._slp_bank_count + 1)
            inserted += 1

        return inserted, replaced, skipped

    def _get_slp_score_delta(self, score, feat_len):
        """Estimate MineCLIP progress inside a replay sequence for landmark priority."""
        try:
            score_seq = score.reshape(-1)
            # If score came from [B, T], use temporal delta before flattening when possible.
            # Otherwise this falls back to a safe zero-delta vector.
            # The caller only uses this as an optional priority term.
            return torch.zeros(feat_len, device=self._config.device, dtype=torch.float32)
        except Exception:
            return torch.zeros(feat_len, device=self._config.device, dtype=torch.float32)

    def _update_slp_landmarks(self, feat, data):
        """Mine strategic, diverse latent landmarks using MineCLIP score.

        MineCLIP is not used as an online dense reward here. It only ranks replay
        states. The actor receives a model-side latent progress reward for moving
        its imagined states toward selected landmarks.
        """
        if not getattr(self._config, "use_slp_reward", False):
            return {}
        with torch.no_grad():
            score = self._get_slp_scores_from_batch(data)
            if score is None:
                return {"slp_bank_size": float(self._slp_bank_count), "slp_no_score_key": 1.0}

            feat = feat.detach().float().reshape(-1, feat.shape[-1])
            raw_score = score.detach().float()
            score = raw_score.reshape(-1).float()
            n = min(feat.shape[0], score.shape[0])
            if n <= 0:
                return {"slp_bank_size": float(self._slp_bank_count)}
            feat, score = feat[:n], score[:n]
            valid = torch.isfinite(score)
            min_score = getattr(self._config, "slp_min_score", None)
            if min_score is not None:
                valid = valid & (score >= float(min_score))
            if not valid.any():
                return {
                    "slp_bank_size": float(self._slp_bank_count),
                    "slp_batch_score_mean": _metric_to_float(score),
                    "slp_selected_num": 0.0,
                }

            feat, score = feat[valid], score[valid]
            frac = float(getattr(self._config, "slp_top_percentile", 0.2))
            frac = max(0.0, min(1.0, frac))
            k = max(1, int(round(feat.shape[0] * frac)))
            k = min(k, int(getattr(self._config, "slp_max_new_landmarks", 128)), feat.shape[0])
            top_score, top_idx = torch.topk(score, k=k, largest=True)
            top_feat = feat[top_idx]
            if getattr(self._config, "slp_distance", "cosine") == "cosine":
                top_feat = torch.nn.functional.normalize(top_feat, dim=-1, eps=1e-6)

            inserted, replaced, skipped = self._insert_slp_landmarks_diverse(top_feat, top_score)
            return {
                "slp_bank_size": float(self._slp_bank_count),
                "slp_selected_num": float(k),
                "slp_inserted_num": float(inserted),
                "slp_replaced_num": float(replaced),
                "slp_skipped_duplicate_num": float(skipped),
                "slp_batch_score_mean": _metric_to_float(score),
                "slp_selected_score_mean": _metric_to_float(top_score),
                "slp_selected_score_max": _metric_to_float(top_score.max()),
            }

    def _strategic_latent_progress_reward(self, imag_feat):
        """Dense progress reward toward reachable strategic MineCLIP landmarks.

        For each imagined sequence, CORE2-SLP selects a small set of landmarks
        that are both semantically strong and reachable from the current imagined
        start state, then rewards increases in similarity to those landmarks.
        """
        scale = self._slp_scale()
        zeros = torch.zeros(imag_feat.shape[:-1] + (1,), device=imag_feat.device, dtype=imag_feat.dtype)
        self._slp_last_metrics = {
            "slp_scale": float(scale),
            "slp_bank_size": float(self._slp_bank_count),
        }
        if scale <= 0.0 or self._slp_landmarks is None or self._slp_bank_count <= 0 or imag_feat.shape[0] < 2:
            return zeros

        with torch.no_grad():
            valid_count = int(self._slp_bank_count)
            landmarks = self._slp_landmarks[:valid_count].detach().float()
            scores = self._slp_scores[:valid_count].detach().float()
            feat = imag_feat.detach().float()

            if getattr(self._config, "slp_distance", "cosine") == "cosine":
                feat_n = torch.nn.functional.normalize(feat, dim=-1, eps=1e-6)
                land_n = torch.nn.functional.normalize(landmarks, dim=-1, eps=1e-6)
                start_sim = torch.matmul(feat_n[0], land_n.t())  # [N, M]
            else:
                feat_n = feat
                land_n = landmarks
                start_sim = -torch.cdist(feat[0], landmarks, p=2) ** 2

            # Normalize MineCLIP scores inside the current landmark bank.
            if scores.numel() > 1:
                score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
            else:
                score_norm = torch.zeros_like(scores)

            # Strategic landmark selection: semantic score minus reachability cost.
            alpha = float(getattr(self._config, "slp_score_priority", 1.0))
            beta = float(getattr(self._config, "slp_reachability_weight", 0.7))
            priority = alpha * score_norm.unsqueeze(0) + beta * start_sim
            cand_k = min(int(getattr(self._config, "slp_candidate_k", 8)), valid_count)
            _, cand_idx = torch.topk(priority, k=cand_k, dim=-1, largest=True)  # [N, K]
            cand = land_n[cand_idx]

            if getattr(self._config, "slp_distance", "cosine") == "cosine":
                sim = torch.einsum("lnd,nkd->lnk", feat_n, cand)
                phi = sim.max(dim=-1).values  # [L, N]
                dist = 1.0 - phi
            else:
                diff = feat.unsqueeze(2) - cand.unsqueeze(0)
                dist_all = torch.sum(diff * diff, dim=-1)
                dist = dist_all.min(dim=-1).values
                phi = -dist

            progress = phi[1:] - phi[:-1]
            if getattr(self._config, "slp_use_positive_only", True):
                progress = torch.relu(progress)
            clip = float(getattr(self._config, "slp_reward_clip", 1.0))
            if clip > 0.0:
                if getattr(self._config, "slp_use_positive_only", True):
                    progress = progress.clamp(0.0, clip)
                else:
                    progress = progress.clamp(-clip, clip)
            reward = torch.zeros_like(zeros)
            reward[1:, :, 0] = progress.to(reward.dtype)
            reward = reward * float(scale)

            selected_scores = torch.gather(score_norm.unsqueeze(0).expand_as(start_sim), 1, cand_idx)
            selected_start_sim = torch.gather(start_sim, 1, cand_idx)
            self._slp_last_metrics = {
                "slp_scale": float(scale),
                "slp_bank_size": float(valid_count),
                "slp_reward_mean": _metric_to_float(reward),
                "slp_reward_max": _metric_to_float(reward.max()),
                "slp_distance_mean": _metric_to_float(dist),
                "slp_phi_mean": _metric_to_float(phi),
                "slp_progress_mean": _metric_to_float(progress),
                "slp_candidate_k": float(cand_k),
                "slp_candidate_score_mean": _metric_to_float(selected_scores.mean()),
                "slp_candidate_start_sim_mean": _metric_to_float(selected_start_sim.mean()),
            }
            return reward

    def _latent_goal_progress_reward(self, imag_feat):
        # Backward-compatible alias used by older call sites.
        return self._strategic_latent_progress_reward(imag_feat)

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, post_zoomed, context, mets = self._wm._train(data)
        metrics.update(mets)
        if getattr(self._config, "use_slp_reward", False):
            metrics.update(self._update_slp_landmarks(context["feat"], data))
        # start = (post, post_zoomed)

        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        def intrinsic(f, s, a):
            # CORE2-SLP: original wrapper-provided intrinsic reward is disabled.
            return torch.zeros(f.shape[:-1] + (1,), device=f.device, dtype=f.dtype)

        def slp_objective(f, s, a):
            reward = self._latent_goal_progress_reward(f)
            slp_objective.last_metrics = self._slp_last_metrics
            return reward
        slp_objective.last_metrics = {}

        jumping_steps = lambda f, s, a: self._wm.heads["jumping_steps"](
            f
        ).mean().clamp_min(1).int()

        accumulated_reward = lambda f, s, a: self._wm.heads["accumulated_reward"](
            f
        ).mode()

        jump_indicator = lambda s: self._wm.heads["jump"](
            self._wm.dynamics.get_feat(s)
        ).mean

        is_end = lambda s: self._wm.heads["end"](
            self._wm.dynamics.get_feat(s)
        ).mean

        metrics.update(self._task_behavior._train(post, post_zoomed, reward, intrinsic, jumping_steps, accumulated_reward, jump_indicator, is_end, slp_objective)[-1])
        if getattr(self._config, "use_slp_reward", False):
            metrics.update(getattr(slp_objective, "last_metrics", {}))
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(post, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "minedojo":
        import envs.minedojo as minedojo
        log_dir = os.path.join(config.results_dir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        kwargs=dict(
                log_dir=log_dir,
                target_item=config.target_item
            )
        env = minedojo.make_env(task, **kwargs)
        env = wrappers.OneHotAction(env)

    else:
        raise NotImplementedError(suite)
    
    # env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)

    return env


def main(config): # config is namespace

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    # 增加一个判断：如果传入的 logdir 中还不包含 seed_，说明是新开训练，生成新路径；
    # 否则说明用户直接传入了旧的时间戳断点目录，直接使用即可。
    if "seed_" not in str(logdir): 
        logdir = logdir / config.task
        logdir = logdir / 'seed_{}'.format(config.seed)
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        logdir = logdir / timestamp
        
    config.logdir = logdir
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat 
    config.log_every //= config.action_repeat 
    config.time_limit //= config.action_repeat
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    
    logger = tools.Logger(config, logdir, config.action_repeat * step)

    if config.offline_traindir: # False
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir

    train_eps = tools.load_episodes(directory, limit=config.dataset_size)

    if config.offline_evaldir: # False
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir

    eval_eps = tools.load_episodes(directory, limit=1)

    make = lambda mode, id: make_env(config, mode, id)
    suite, task = config.task.split("_", 1)
    
    from envs.tasks import get_specs

    kwargs=dict(
            # log_dir=log_dir,
            target_item=config.target_item
        )
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict

    config.episode_max_steps = task_specs['terminal_specs']['max_steps']
    task_specs['concentration_specs']['max_steps'] = task_specs['terminal_specs']['max_steps']
    task_specs['concentration_specs']['gaussian_reward_weight'] = config.gaussian_reward_weight
    task_specs['concentration_specs']['gaussian_sigma_weight'] = config.gaussian_sigma_weight
    task_specs['clip_specs']['target_object'] = task_specs['success_specs']['all']['item']['type'] if 'all' in task_specs['success_specs'] else task_specs['success_specs']['any']['item']['type']
    
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    step_calculator = tools.ScoreStorage(max_steps=config.episode_max_steps)

    state = None

    if not config.offline_traindir: 
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=prefill,
            is_training=False,
        )

        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = LS_Imagine(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)

    agent.requires_grad_(requires_grad=False)
    
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every: 
        logger.write()
        
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False) 
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                step_calculator,
                config.episode_max_steps,
                config.discount,
                is_eval=True,
                episodes=config.eval_episode_num,
                is_training=False,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        print("Start training.")

        state = tools.simulate(
            agent, # LS_Imagine
            train_envs, 
            train_eps,
            config.traindir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=config.eval_every, 
            state=state,
            is_training=True,
        )

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name]) 

    parser = argparse.ArgumentParser()

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))
