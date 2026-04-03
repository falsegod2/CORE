import os
import cv2
import gym
import numpy as np
import habitat

# 保持日志清爽，去掉强制指定 EGL_DEVICE_ID，避免底层驱动进行双重设备映射从而产生冲突
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

class HabitatDreamerWrapper(gym.Env):
    def __init__(self, config_path, res=(64, 64)):
        # 1. 获取基础配置
        config = habitat.get_config(config_path)
        
        # 2. 【关键修改】动态修改配置，解决 EGL 与 CUDA 设备的绑定冲突
        with habitat.config.read_write(config):
            # 强制指定底层渲染器使用 GPU 0
            # 注意：如果等下运行依然报 unable to find CUDA device，请尝试把这里的 0 改成 -1 (代表禁用CUDA映射)
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1 
            
        # 3. 初始化环境
        self._env = habitat.Env(config=config)
        self._res = res
        
        # Habitat PointNav 通常有 4 个离散动作: 0:STOP, 1:FORWARD, 2:LEFT, 3:RIGHT
        self.action_space = gym.spaces.Discrete(4)
        
        # Dreamer 要求的观测空间字典格式
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, self._res + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        })

    def _process_obs(self, obs, reward, is_first, is_last, is_terminal):
        # 提取 RGB 图像并 Resize 到 Dreamer 指定大小
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
        
        # 从 Habitat metrics 中获取 reward 和 done
        metrics = self._env.get_metrics()
        reward = metrics.get('reward', 0.0) 
        done = self._env.episode_over
        
        return self._process_obs(obs, reward, False, done, done)

    def reset(self):
        obs = self._env.reset()
        return self._process_obs(obs, 0.0, True, False, False)

    def close(self):
        self._env.close()

if __name__ == "__main__":
    print("正在初始化 Habitat 无头环境...")
    
    # 绝对路径配置
    test_config = "/gz-data/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    
    try:
        env = HabitatDreamerWrapper(test_config, res=(64, 64))
        
        print("重置环境并获取第一帧观测...")
        obs = env.reset()
        
        print("观测字典包含的键:", obs.keys())
        print("图像尺寸:", obs['image'].shape)
        
        # 将 RGB 转换为 BGR 并保存第一帧
        image_bgr = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite("habitat_first_frame.png", image_bgr)
        
        # 走几步测试
        for i in range(5):
            obs = env.step(1) # 1 通常是前进
        
        # 保存第五帧
        image_bgr_step5 = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite("habitat_step5_frame.png", image_bgr_step5)
        
        print("测试完毕！已保存 habitat_first_frame.png 和 habitat_step5_frame.png。")
        env.close()
        
    except Exception as e:
        print(f"初始化失败，报错信息: {e}")