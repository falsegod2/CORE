from gym import Wrapper

class ConcentrationWrapper(Wrapper):
    def __init__(self, env, concentration, prompts=None, dense_reward=0.01,
                 mineclip_dense_reward=0.01, max_steps=1000,
                 gaussian_reward_weight=1.0, score_quantum=0.0,
                 improvement_eps=0.0, **kwargs):
        super().__init__(env)
        self.concentration = concentration # ConcentrationReward
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.mineclip_dense_reward = mineclip_dense_reward
        self.gaussian_reward_weight = gaussian_reward_weight
        self.score_quantum = float(score_quantum)
        self.improvement_eps = float(improvement_eps)

        self.episode = 0
        self.steps = 0
        self.last_score = 0

        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        self.max_steps = max_steps

    def reset_reproducibility_state(self):
        self.concentration.reset_running_statistics()
        self.episode = 0
        self.steps = 0
        self.last_score = 0
        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

    def _q(self, value):
        value = float(value)
        if self.score_quantum <= 0:
            return value
        return round(value / self.score_quantum) * self.score_quantum

    def reset(self, **kwargs):
        self.episode += 1
        self.steps = 0

        self.last_score = 0
        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0
        obs = self.env.reset(**kwargs)

        score, zoom_in_prob, check_threshold = self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
        zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
        if is_check:
            mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
        else:
            mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = 0.0, 0.0, 0.0, False, False

        score = self._q(score)
        mineclip_on_zoomed = self._q(mineclip_on_zoomed)
        gaussian_on_zoomed = self._q(gaussian_on_zoomed)
        zoom_in_prob_on_zoomed = self._q(zoom_in_prob_on_zoomed)

        obs['is_zoomed'] = is_zoomed
        obs['jump'] = jump
        obs['jumping_steps'] = self.max_steps
        obs['accumulated_reward'] = 0.0
        obs['is_calculated'] = False
        obs['reward_on_zoomed'] = 0.0
        obs['intrinsic_on_zoomed'] = 0.0
        obs['score_on_zoomed'] = 0.0
        obs['zoomed_image'] = zoomed_image

        if score > self.last_score + self.improvement_eps:
            obs['intrinsic'] += self.dense_reward * score * self.gaussian_reward_weight
            self.last_score = score

        obs['score'] += self.dense_reward * score

        if is_zoomed:
            if gaussian_on_zoomed > self.last_score + self.improvement_eps and gaussian_on_zoomed > self.last_zoom_in_gaussian_score + self.improvement_eps:
                obs['intrinsic_on_zoomed'] += self.dense_reward * gaussian_on_zoomed * self.gaussian_reward_weight
                self.last_zoom_in_gaussian_score = gaussian_on_zoomed

            obs['score_on_zoomed'] += self.dense_reward * gaussian_on_zoomed

            if mineclip_on_zoomed > self.last_zoom_in_mineclip_score + self.improvement_eps:
                obs['intrinsic_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
                self.last_zoom_in_mineclip_score = mineclip_on_zoomed

            obs['score_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
           
        obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)
        if is_zoomed:
            obs['heatmap_on_zoomed'] = self.concentration.get_heatmap(is_zoomed=True)
        else:
            obs['heatmap_on_zoomed'] = obs['heatmap']
        
        return obs
    
    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            score, zoom_in_prob, check_threshold = self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
            zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
            if is_check:
                mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
            else:
                mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = 0.0, 0.0, 0.0, False, False

            score = self._q(score)
            mineclip_on_zoomed = self._q(mineclip_on_zoomed)
            gaussian_on_zoomed = self._q(gaussian_on_zoomed)
            zoom_in_prob_on_zoomed = self._q(zoom_in_prob_on_zoomed)
            
            obs['is_zoomed'] = is_zoomed
            obs['jump'] = jump
            obs['jumping_steps'] = self.max_steps
            obs['accumulated_reward'] = 0.0
            obs['is_calculated'] = False
            obs['reward_on_zoomed'] = reward
            obs['intrinsic_on_zoomed'] = 0.0
            obs['score_on_zoomed'] = 0.0
            obs['zoomed_image'] = zoomed_image

            if score > self.last_score + self.improvement_eps:
                obs['intrinsic'] += self.dense_reward * score * self.gaussian_reward_weight
                self.last_score = score

            obs['score'] += self.dense_reward * score

            if is_zoomed:
                if gaussian_on_zoomed > self.last_score + self.improvement_eps and gaussian_on_zoomed > self.last_zoom_in_gaussian_score + self.improvement_eps:
                    obs['intrinsic_on_zoomed'] += self.dense_reward * gaussian_on_zoomed * self.gaussian_reward_weight
                    self.last_zoom_in_gaussian_score = gaussian_on_zoomed

                obs['score_on_zoomed'] += self.dense_reward * gaussian_on_zoomed
                
                if mineclip_on_zoomed > info["clip_last_score"] + self.improvement_eps and mineclip_on_zoomed > self.last_zoom_in_mineclip_score + self.improvement_eps:
                    self.mineclip_dense_reward = info["clip_dense_reward"]
                    obs['intrinsic_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
                    self.last_zoom_in_mineclip_score = mineclip_on_zoomed

                obs['score_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed

            obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)
            if is_zoomed:
                obs['heatmap_on_zoomed'] = self.concentration.get_heatmap(is_zoomed=True)
            else:
                obs['heatmap_on_zoomed'] = obs['heatmap']
                
        return obs, reward, done, info
        


    

                
