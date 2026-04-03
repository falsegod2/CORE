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
        
        # 动态修改 Config，使用 -1 禁用严格的 CUDA 设备匹配
        config = habitat.get_config(config_path)
        with habitat.config.read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1 
            
        self._env = habitat.Env(config=config)
        self._res = res
        
        # Habitat 离散动作空间
        self.action_space = gym.spaces.Discrete(4)
        
        # Dreamer 字典观测空间 (注意：移除了 reward，因为 Gym 会将其作为单独返回值)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, self._res + (3,), dtype=np.uint8),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        })

    def _process_obs(self, obs, is_first, is_last, is_terminal):
        image = obs['rgb']
        if image.shape[:2] != self._res:
            image = cv2.resize(image, self._res, interpolation=cv2.INTER_AREA)
            
        return {
            'image': image,
            'is_first': np.bool_(is_first),
            'is_last': np.bool_(is_last),
            'is_terminal': np.bool_(is_terminal),
        }

    def step(self, action):
        obs = self._env.step(action)
        
        metrics = self._env.get_metrics()
        # 获取奖励
        reward = metrics.get('reward', 0.0) 
        done = self._env.episode_over
        
        # 获取观测字典
        dict_obs = self._process_obs(obs, False, done, done)
        
        # 【关键修复】严格返回 4 个值：obs, reward, done, info
        return dict_obs, float(reward), done, {}

    def reset(self):
        obs = self._env.reset()
        # 【关键修复】经典 Gym 的 reset 只返回 obs
        dict_obs = self._process_obs(obs, True, False, False)
        return dict_obs

    def close(self):
        self._env.close()