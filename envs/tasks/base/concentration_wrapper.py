from gym import Wrapper


class ConcentrationWrapper(Wrapper):
    def __init__(
        self,
        env,
        concentration,
        prompts=None,
        dense_reward=0.01,
        mineclip_dense_reward=0.01,
        max_steps=1000,
        gaussian_reward_weight=1.0,
        disable_intrinsic=False,
        disable_long_branch=False,
        **kwargs,
    ):
        super().__init__(env)
        self.concentration = concentration  # ConcentrationReward
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.mineclip_dense_reward = mineclip_dense_reward
        self.gaussian_reward_weight = gaussian_reward_weight
        self.disable_intrinsic = disable_intrinsic
        self.disable_long_branch = disable_long_branch

        self.episode = 0
        self.steps = 0
        self.last_score = 0

        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        self.max_steps = max_steps

    def reset(self, **kwargs):
        self.episode += 1
        self.steps = 0

        self.last_score = 0
        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0
        obs = self.env.reset(**kwargs)

        if len(self.prompt) > 0:
            score, _, _ = self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
        else:
            score = 0.0

        self._apply_state_score(obs, score)
        obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)

        if self.disable_long_branch:
            self._attach_no_long_fields(obs, reward_on_zoomed=0.0)
            return obs

        zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
        if is_check:
            mineclip_on_zoomed, gaussian_on_zoomed, _, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
        else:
            mineclip_on_zoomed, gaussian_on_zoomed, is_zoomed, jump = 0.0, 0.0, False, False

        self._attach_long_fields(
            obs,
            reward_on_zoomed=0.0,
            zoomed_image=zoomed_image,
            is_zoomed=is_zoomed,
            jump=jump,
            mineclip_on_zoomed=mineclip_on_zoomed,
            gaussian_on_zoomed=gaussian_on_zoomed,
            clip_last_score=0.0,
            clip_dense_reward=self.mineclip_dense_reward,
        )
        return obs

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            score, _, _ = self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
        else:
            score = 0.0

        self._apply_state_score(obs, score)
        obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)

        if self.disable_long_branch:
            self._attach_no_long_fields(obs, reward_on_zoomed=reward)
            return obs, reward, done, info

        zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
        if is_check:
            mineclip_on_zoomed, gaussian_on_zoomed, _, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
        else:
            mineclip_on_zoomed, gaussian_on_zoomed, is_zoomed, jump = 0.0, 0.0, False, False

        self._attach_long_fields(
            obs,
            reward_on_zoomed=reward,
            zoomed_image=zoomed_image,
            is_zoomed=is_zoomed,
            jump=jump,
            mineclip_on_zoomed=mineclip_on_zoomed,
            gaussian_on_zoomed=gaussian_on_zoomed,
            clip_last_score=info.get("clip_last_score", 0.0),
            clip_dense_reward=info.get("clip_dense_reward", self.mineclip_dense_reward),
        )
        return obs, reward, done, info

    def _apply_state_score(self, obs, score):
        """Keep score for diagnostics, but optionally remove intrinsic reward shaping."""
        if 'intrinsic' not in obs:
            obs['intrinsic'] = 0.0
        if 'score' not in obs:
            obs['score'] = 0.0

        if score > self.last_score:
            if not self.disable_intrinsic:
                obs['intrinsic'] += self.dense_reward * score * self.gaussian_reward_weight
            self.last_score = score

        # score is a logged/auxiliary signal. It is not added to env reward here.
        obs['score'] += self.dense_reward * score

    def _attach_no_long_fields(self, obs, reward_on_zoomed=0.0):
        """Attach neutral LS-Imagine fields for a clean no-long baseline."""
        obs['is_zoomed'] = False
        obs['jump'] = False
        obs['jumping_steps'] = 1.0
        obs['accumulated_reward'] = 0.0
        obs['is_calculated'] = True
        obs['reward_on_zoomed'] = reward_on_zoomed
        obs['intrinsic_on_zoomed'] = 0.0
        obs['score_on_zoomed'] = 0.0
        obs['heatmap_on_zoomed'] = obs['heatmap']
        # Do not set zoomed_image. LSImagineWrapper will insert a zero image.

    def _attach_long_fields(
        self,
        obs,
        reward_on_zoomed,
        zoomed_image,
        is_zoomed,
        jump,
        mineclip_on_zoomed,
        gaussian_on_zoomed,
        clip_last_score,
        clip_dense_reward,
    ):
        obs['is_zoomed'] = is_zoomed
        obs['jump'] = jump
        obs['jumping_steps'] = self.max_steps
        obs['accumulated_reward'] = 0.0
        obs['is_calculated'] = False
        obs['reward_on_zoomed'] = reward_on_zoomed
        obs['intrinsic_on_zoomed'] = 0.0
        obs['score_on_zoomed'] = 0.0
        obs['zoomed_image'] = zoomed_image

        if is_zoomed:
            if gaussian_on_zoomed > self.last_score and gaussian_on_zoomed > self.last_zoom_in_gaussian_score:
                if not self.disable_intrinsic:
                    obs['intrinsic_on_zoomed'] += self.dense_reward * gaussian_on_zoomed * self.gaussian_reward_weight
                self.last_zoom_in_gaussian_score = gaussian_on_zoomed

            obs['score_on_zoomed'] += self.dense_reward * gaussian_on_zoomed

            if mineclip_on_zoomed > clip_last_score and mineclip_on_zoomed > self.last_zoom_in_mineclip_score:
                self.mineclip_dense_reward = clip_dense_reward
                if not self.disable_intrinsic:
                    obs['intrinsic_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
                self.last_zoom_in_mineclip_score = mineclip_on_zoomed

            obs['score_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
            obs['heatmap_on_zoomed'] = self.concentration.get_heatmap(is_zoomed=True)
        else:
            obs['heatmap_on_zoomed'] = obs['heatmap']
