from gym import Wrapper

class ConcentrationWrapper(Wrapper):
    def __init__(self, env, concentration, prompts=None, dense_reward=0.01, mineclip_dense_reward=0.01, max_steps=1000, gaussian_reward_weight=1.0, **kwargs):
        super().__init__(env)
        self.concentration = concentration # ConcentrationReward
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.mineclip_dense_reward = mineclip_dense_reward
        self.gaussian_reward_weight = gaussian_reward_weight

        self.episode = 0
        self.steps = 0
        self.max_steps = max_steps

    def reset(self, **kwargs):
        self.episode += 1
        self.steps = 0

        obs = self.env.reset(**kwargs)

        if len(self.prompt) > 0:
            self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
            obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)

        return obs
    
    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
            obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)
                
        return obs, reward, done, info
