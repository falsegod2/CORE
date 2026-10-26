# Clean LS + Evidence 消融实验（2026-10-25）

本包在 `LS_Imagine_Evidence_Ablations_20261021` 上新增两组无记忆组合消融、固定回放诊断和 LS/no_memory 配对种子入口。训练损失权重与原 Clean 环境协议不变。正式实验请以本文和 [诊断及配对运行说明](README_DIAGNOSTICS_PAIRED_CN.md) 为准；旧说明保留用于追溯历史版本。

## 1. 安装位置与已有资源

解压到新的目录，保留原实验目录：

```bash
cd "$HOME/CJJ"
unzip -n LS_Imagine_Evidence_Ablations_20261025.zip
cd LS_Imagine_Evidence_Ablations_20261025
conda activate ls_imagine
```

脚本优先使用 `$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python`；如不同，设置 `ENV_PYTHON=/你的环境/bin/python`。包内不包含 MineCLIP/U-Net 权重、Java 或 MineDojo 安装，不会修改你现有的 Gradle 缓存。

`--assets-root` 应指向同时包含 `weights/mineclip_attn.pth` 和任务 `affordance_map/.../swin_unet_checkpoint.pth` 的目录。也可用 `CORE_ASSETS` 指定；自动查找包含之前的 `$HOME/CJJ/LS_Imagine_Clean_Evidence_20261020`。本地 tokenizer 如未被找到，可设置：

```bash
export MINECLIP_TOKENIZER=/你的已有目录/clip-vit-base-patch16
export JAVA_HOME=/你的原来可运行的JDK8目录
```

`JAVA_HOME` 仅在自动检测失败时需要设置。所有示例中的 GPU1 都是物理选择 `--gpu 1`。入口在导入 PyTorch/teacher 前设置 CUDA 可见设备；模型内的 `cuda:0` 表示唯一可见的 GPU1。训练和评估在同一个可见设备配置下串行执行。

## 2. 消融组

| `--ablation` | 相对完整 Evidence 的变化 | 保留内容 |
|---|---|---|
| `full` | 完整方法，对照组 | 观察记忆、S/Z 动力学、FC、RC、语义监督 |
| `no_memory` | 关闭记忆 token 的写入/读取路径 | 局部语义预测辅助头、FC、RC；不是把全部语义监督一起删掉 |
| `no_memory_no_rc` | 关闭记忆和 RC | FC、局部/后验语义监督、task/clip 奖励头 |
| `no_memory_no_fc_rc` | 关闭记忆、FC 和 RC | 局部/后验语义监督、task/clip 奖励头 |
| `memory_start_only` | 想象 rollout 仅在起点读取记忆 | 真实观察的记忆积累、其余辅助损失 |
| `no_fc` | 关闭 FC | 记忆、局部/后验语义监督、RC |
| `no_rc` | 关闭 RC | 记忆、语义监督、FC |
| `no_fc_rc` | 同时关闭 FC、RC | 记忆、局部/后验语义监督 |
| `no_semantic` | 关闭局部热图、后验热图和 FC 热图项；不加载 U-Net teacher | RGB 记忆、FC 的非热图项、RC、MineCLIP 奖励 |

`no_semantic` 是“去语义监督”的组合消融。`no_memory` 中共享的 RGB 特征/局部热图预测仍可学习，但不向动力学提供记忆。这里没有把 `no_semantic` 误作“去 MineCLIP 奖励”。

查看实际配置（不启动环境）：

```bash
bash run_experiment.sh --method evidence_dreamer --list-ablations
bash run_experiment.sh --method evidence_dreamer --ablation no_fc --gpu 1 --seed 1 --dry-run
bash run_experiment.sh --method ls_imagine --gpu 1 --seed 1 --dry-run
```

`--set PATH=VALUE` 可用于显式超参数调整。做标准消融时建议直接使用命名组并保持其余参数一致；不要在同一组的不同 seed 上改变奖励、训练预算或评估口径。

## 3. 单个实验

完整 Evidence，GPU1、seed1：

```bash
bash run_experiment.sh --method evidence_dreamer --ablation full --gpu 1 --seed 1 --check
bash run_experiment.sh --method evidence_dreamer --ablation full --gpu 1 --seed 1 --steps 1000000
```

去 RC 消融，GPU1、seed1：

```bash
bash run_experiment.sh --method evidence_dreamer --ablation no_rc --gpu 1 --seed 1 --steps 1000000
```

原 LS-Imagine 算法，GPU1、seed1：

```bash
bash run_experiment.sh --method ls_imagine --gpu 1 --seed 1 --check
bash run_experiment.sh --method ls_imagine --gpu 1 --seed 1 --steps 1000000
```

`--check` 检查已有资源/依赖/CUDA/tokenizer，不启动 Minecraft。需要先实际走一次训练和评估重置时，可用 `--stage smoke`，会使用小网络和独立 smoke 输出目录；smoke 结果不能用于正式方法效果比较。

LS 保留原有 heatmap 输入、concentration shaping、zoom 及长时想象分支。Evidence 保留 RGB policy、只作训练目标的语义标签，以及自身 FC/RC。两种方法共享 Clean 的任务、HUD、动作、终止和 SemifastResetWrapper，不加入 WM 的 RPC/异步评估或固定世界库。两种方法的信息利用和辅助奖励机制仍然不同，不能仅凭“同框架”声称信息预算相同。

九个 Evidence 组在同 seed 下，共有世界模型、actor、value 和 slow value 的初始化参数一致。这不表示 LS 与 Evidence 网络初始化相同，也不保证不同方法逐回合遇到相同世界。两个入口均关闭 `torch.compile`。LS 的总训练步数包含原算法收集长期标签所需的尾段环境动作；评估动作不计入训练预算。

## 4. 输出与恢复

默认目录由方法、消融组、stage、task 和 seed 区分；`--dry-run` 中的 `logdir` 是准确位置。也可用 `--logdir /绝对路径` 指定独立目录。已有实验未加 `--resume` 会被拒绝，避免覆盖；同一目录也不能由两个训练进程同时持有。

恢复时使用同一个实验的参数并增加 `--resume`：

```bash
bash run_experiment.sh --method evidence_dreamer --ablation no_rc --gpu 1 --seed 1 --steps 1000000 --resume
bash run_experiment.sh --method ls_imagine --gpu 1 --seed 1 --steps 1000000 --resume
```

正式配置默认每 20,000 个累计训练动作保存一次，保留最近两份完整 checkpoint；不是从 100,000 重新计数。`--steps` 是累计目标，恢复按最新完整 checkpoint 的实际步数继续。保存模型、优化器、replay 和训练调度；Minecraft 物理世界需要重新开始一个 episode，不能逐帧接续。日志追加保留，checkpoint 之后尚未保存的交互可能重做。

源码、关键配置与资源身份不匹配会拒绝恢复。**本包新增功能后的实验应使用新目录；不要覆盖仍在运行的 20261021 代码，不要直接拿 20261021、20261020、旧 WM 或原 `expr.py` 的 checkpoint 强行恢复。** 原实验继续用原源码、原命令和原目录恢复；新诊断/新组合从本包的新目录开始。不要删除或修改 `provenance.json` 来绕过身份检查。

在 tmux 会话中启动后，可以断开本地连接：

```bash
tmux new -s evidence_no_rc_seed1
# 在会话内执行训练命令；需要分离时按 Ctrl+B，松开，再按 D。
# 重新连接后：tmux attach -t evidence_no_rc_seed1
```

## 5. 多 seed 消融计划

先打印完整命令计划；此命令不会启动训练：

```bash
"${ENV_PYTHON:-$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python}" run_ablation_suite.py \
  --gpu 1 --seeds 0 1 2 --include-ls
```

确认资源和单次 smoke 运行正常后，加 `--execute` 才会依次运行全部实验。默认 9 个 Evidence 组 × 3 个 seed；`--include-ls` 再增加 3 个 LS 实验，共 30 次，各自独立目录。不会在一个 GPU 上同时开 30 个 Minecraft；任一实验失败即停止队列。

只运行部分组可用 `--ablations full no_memory no_fc no_rc`。队列的 `--resume` 要求其中每个实验已有有效 checkpoint；部分实验从未启动时，分别使用单次实验命令启动/恢复。

本次最直接的配对计划使用独立的 `paired_20261025` 目录：

```bash
bash run_paired_benchmark.sh --gpu 1 --seeds 0 1 2
# 上一行只打印计划；加 --execute 才启动。
bash run_paired_benchmark.sh --gpu 1 --seeds 0 1 2 --execute
```

默认仅 LS 与 no_memory 两方法、三个种子，共六次。新增组合可用 `--methods no_memory_no_rc no_memory_no_fc_rc`。分 GPU、恢复和只读统计的完整命令见 [README_DIAGNOSTICS_PAIRED_CN.md](README_DIAGNOSTICS_PAIRED_CN.md)。

## 6. 日志与解释

Evidence 新增真实训练/评估动作计数和频率，以及想象中的 task/clip 奖励分项。可区分“总想象奖励变高”是哪个 head 的变化，并检查真实动作是否长期偏向 action3（向下看）。

具体字段为 `imag_task_reward_mean`、`imag_clip_reward_mean`、`imag_scaled_clip_reward_mean`，以及 `real_action_counts` / `real_action_fractions`、`eval_action_counts` / `eval_action_fractions`。本版 LS 的 `metrics.jsonl` 同时含原有标量和运行事件；`runtime.jsonl` 保留兼容副本。LS 的结构化 `evaluation` 事件把重复摘要放在 `result` 内，统计评估曲线只读取顶层 `eval_success`，避免重复计数。

RC 条件误差改为按日志窗口累计误差分子/有效权重，再求比值；没有正样本时记为 `null`，不会再用 0 稀释正样本 MAE。此修正改变诊断口径，不改变完整 Evidence 的 RC 训练损失；新旧 RC MAE 曲线不能直接当成同一统计量。

本版每 20,000 步额外记录 `event=diagnostics` 的固定回放诊断；数据和采样位置保存进 checkpoint。诊断测量零/非零目标回报误差、FC/RC 梯度与 Memory 读取，不更新参数。`train` 行仍然是原训练采样窗口的指标，不能与固定 bank 指标混为同一条曲线。字段、限制和开关见新诊断说明。

`task_reward` 始终保留底层原生任务奖励；success 独立读取环境成功标志，不能把 reward 恒当作严格二值成功标签。MineCLIP record-high intrinsic 延续 Clean 实现，不是相邻帧分数差。

验证范围和可复现的测试结果见 `verification/ABLATIONS_VALIDATION.md`。本地 CPU 测试不等于你的远程 Minecraft/Java/Gradle 启动与正式 GPU 百万步实验已经验证。
