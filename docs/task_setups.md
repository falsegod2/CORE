# How to Set Up Tasks in MineDojo

We have predefined the task settings mentioned in the [paper](https://arxiv.org/pdf/2410.03618) in the `./envs/tasks/task_specs.yaml` file. Each task includes configurations for three stages: data collection, world model and behavior learning, and testing. For example, the task ***Harvest log in plains*** is defined as `collect_rollouts_harvest_log_in_plains`, `harvest_log_in_plains`, and `test_harvest_log_in_plains` for these three stages, respectively. 

This document introduces the meaning and configuration guidelines for each field in the task settings, enabling you to define your custom tasks by following these examples.

## Field Descriptions and Guidelines

- **`task_id`**: Generally set to `harvest`. For more details, refer to [Task Customization](https://docs.minedojo.org/sections/customization/task.html#task-customization).
- **`sim`**: Typically set to `minedojo`.
- **`fast_reset`**: For more information, refer to [Reset Mode](https://docs.minedojo.org/sections/customization/task.html#reset-mode).  
  - During world model and behavior learning, `fast_reset` is often set to `5` or `10` to stabilize the training process.  
  - During data collection and testing, `fast_reset` is usually set to `0` to collect more diverse data.

- **`sim_specs`**: Details of the environment setup. See [Task Customization](https://docs.minedojo.org/sections/customization/task.html#task-customization) for more information.
  - **`target_names`**: Specifies the names of the target items for `harvest` tasks.
  - **`target_quantities`**: Specifies the quantities of the target items for `harvest` tasks.
  - **`specified_biome`**: Defines the initial biome of the environment. See [Specified Biome](https://docs.minedojo.org/sections/customization/sim.html#specified-biome).
  - **`break_speed_multiplier`**: Adjusts the breaking speed of attacks. See [Set Breaking Speed](https://docs.minedojo.org/sections/customization/sim.html#set-breaking-speed). Commonly set to `100` to simplify training.
  - **`initial_inventory`**: Specifies the initial inventory setup. See [Initial Inventory](https://docs.minedojo.org/sections/customization/sim.html#initial-inventory).
  - **`initial_mobs`**: Specifies the initial mobs in the environment. See [Spawn Mobs](https://docs.minedojo.org/sections/customization/sim.html#spawn-mobs).

- **`clip_specs`**: Configuration for using `MineCLIP` to calculate intrinsic rewards.
  - **`prompts`**: Short textual descriptions of the task, specified as `list[str]`.

- **`concentration_specs`**: Configuration for calculating intrinsic rewards based on affordance maps and jumping flags.
  - **`prompts`**: Short textual descriptions of the task, specified as `list[str]`.
  - **`unet_checkpoint_dir`**: Path to the multimodal U-Net weight files.

- **`reward_specs`**: Specifies rewards for collecting different items.

- **`success_specs`**: Configures additional success criteria and actions upon success.
  - **`reward`**: Additional reward upon success.
  - **`any`**: Specifies that the task is considered successful if any of the listed conditions are met.
  - **`all`**: Specifies that the task is considered successful if all listed conditions are met.

- **`terminal_specs`**: Configures conditions for ending an episode.
  - **`max_steps`**: Specifies the maximum number of steps per episode.

- **`screenshot_specs`**: Configuration for saving observation images.
  - **`reset_flag`**: Whether to save images during `reset`. (Images during reset are often darker.)
  - **`step_flag`**: Whether to save images during each `step`.
  - **`save_freq`**: Interval between saved images on average.

- **`LS_Imagine_specs`**: Settings for integrating the LS-Imagine algorithm with the MineDojo environment.
  - **`repeat`**: Number of times a single action is repeated. Typically set to `1`.

  1.当前修改的核心是把你的 no-intrinsic 版 Dreamer / LS-Imagine 代码改成一个 Affordance-guided Object-Centric Dreamer：首先在环境侧恢复 heatmap 输出，并重新启用 MineDojo 的 affordance / concentration wrapper，让 MineCLIP + Swin-Unet 生成的任务相关 affordance map 能进入 observation；然后在模型侧不把 heatmap 当普通图像通道拼进 CNN，而是新增 SlotAttention 和 ObjectCentricConvEncoder，先用 CNN 把 RGB 图像编码成 spatial tokens，再用 affordance map 作为 attention bias，引导 slots 更关注任务相关、可交互区域，最后把多个 object slots 展平成 Dreamer RSSM 的输入 embedding；同时在 preprocess() 中对 heatmap 做归一化，并建议让 decoder 同时重建 image|heatmap，使世界模型 latent 保留任务相关空间信息；此外还加入可选的 MineCLIP score 辅助预测头，并允许在 imagination reward 中按较小权重加入预测的 score，从而让 actor-critic 的 latent imagination 不只依赖稀疏环境奖励，也能利用语言目标相关的进展信号。整体上，这些修改把原来的 monolithic visual latent 改成了由 affordance 引导的 object-centric latent，同时保留 Dreamer 主体结构不变，形成了“MineCLIP/affordance 引导对象中心世界模型”的创新点。

  2.刚刚这轮修改是在当前 Affordance-guided Object-Centric Dreamer 的基础上继续补强闭环：先在 MultiEncoder 和 ObjectCentricConvEncoder 中加入 debug 打印，确认 heatmap 被识别为 affordance_shapes，并且真正以降采样后的 affordance bias 传入 Slot Attention；然后在环境 wrapper 里对 heatmap 增加 np.nan_to_num、resize、clip 等防护，避免 affordance map 中的 NaN/Inf 导致 cast warning；接着把 Slot Attention 的 attn 和降采样后的 aff 保存下来，不再只作为前向 bias，而是新增 affordance_alignment_loss()，让所有 slots 的总体注意力分布显式对齐高 affordance 区域；最后在 WorldModel._train() 中把这个辅助损失按 affordance_align_scale 加进 model loss，并记录 affordance_align_loss 指标。这样修改后，heatmap 不只是被动输入或简单拼接，而是同时作为 attention bias 和辅助监督信号，引导 object slots 主动绑定任务相关、可交互区域。