import gym
import cv2
import numpy as np
import habitat
import os

class HabitatDreamerEnv(gym.Env):
    def __init__(self, config_path, res=(64, 64)):
        os.environ["HABITAT_SIM_LOG"] = "quiet"
        os.environ["MAGNUM_LOG"] = "quiet"
        
        config = habitat.get_config(config_path)
        with habitat.config.read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1 
            
        self._env = habitat.Env(config=config)
        self._res = res
        
        # 【修改1】动作空间从 4 改为 3 (剔除智能体主动调用的 STOP)
        # 我们映射: 0->FORWARD, 1->LEFT, 2->RIGHT
        self.action_space = gym.spaces.Discrete(3)
        self._prev_distance = None
        
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
        # 【修改2】动作映射: Dreamer(0,1,2) -> Habitat(1,2,3)
        # 巧妙地避开了 0 (STOP)
        habitat_action = action + 1 
        
        obs = self._env.step(habitat_action)
        metrics = self._env.get_metrics()
        done = self._env.episode_over
        
        # 【修改3】手工打造稠密奖励 (Dense Reward)
        current_distance = metrics.get('distance_to_goal', None)
        reward = 0.0
        
        if current_distance is not None and self._prev_distance is not None:
            # 每靠近目标 1 米，给 1.0 的奖励；远离则惩罚
            reward = self._prev_distance - current_distance
            
        self._prev_distance = current_distance
        
        # 【修改4】自动停止机制 (Auto-STOP)
        # 如果距离目标极近（通常PointNav阈值是0.2米），强制调用 STOP 判定胜利
        if current_distance is not None and current_distance < 0.2:
            # 给环境发送真正的 STOP 指令以获取最终的 Success Metric
            obs = self._env.step(0) 
            reward += 10.0  # 到达终点，给予巨大奖励！
            done = True
            
        # 增加极其微小的 step 惩罚，鼓励智能体走捷径
        reward -= 0.01 
            
        dict_obs = self._process_obs(obs, False, done, done)
        return dict_obs, float(reward), done, {}

    def reset(self):
        obs = self._env.reset()
        
        # 初始化上一帧距离
        metrics = self._env.get_metrics()
        self._prev_distance = metrics.get('distance_to_goal', None)
        
        return self._process_obs(obs, True, False, False)

    def close(self):
        self._env.close()