import gym
import cv2
import numpy as np
import habitat
import os

class HabitatDreamerEnv(gym.Env):
    def __init__(self, config_path, res=(64, 64)):
        # 屏蔽底层烦人的日志
        os.environ["HABITAT_SIM_LOG"] = "quiet"
        os.environ["MAGNUM_LOG"] = "quiet"
        
        # 动态修改 Config 绑定 GPU 0
        config = habitat.get_config(config_path)
        with habitat.config.read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1 
            
        self._env = habitat.Env(config=config)
        self._res = res
        
        # Habitat 离散动作空间
        self.action_space = gym.spaces.Discrete(4)
        
        # Dreamer 字典观测空间
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, self._res + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        })

    def _process_obs(self, obs, reward, is_first, is_last, is_terminal):
        image = obs['rgb']
        if image.shape[:2] != self._res:
            image = cv2.resize(image, self._res, interpolation=cv2.INTER_AREA)
            
        return {
            'image': image,
            'reward': np.float32(reward),
            'is_first': np.bool_(is_first),
            'is_last': np.bool_(is_last),
            'is_terminal': np.bool_(is_terminal),
        }

    def step(self, action):
        obs = self._env.step(action)
        
        metrics = self._env.get_metrics()
        # 如果你的基础 config 没有奖励，暂时用 0.0，后续我们可以自定义稠密奖励
        reward = metrics.get('reward', 0.0) 
        done = self._env.episode_over
        
        return self._process_obs(obs, reward, False, done, done)

    def reset(self):
        obs = self._env.reset()
        return self._process_obs(obs, 0.0, True, False, False)

    def close(self):
        self._env.close()