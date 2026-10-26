> 2026-10-21 新入口与消融说明：请先阅读 [README_ABLATIONS_CN.md](README_ABLATIONS_CN.md)。本文下方保留历史版本说明，历史测试数量与启动目录不代表当前包。

# Clean LS 环境上的 EvidenceDreamer

本版本以你上传的 `LS-Imagine_Clean_PixelAffordance_BinaryReward(2).zip` 为基底，接入完整 Evidence 方法。原来的 LS 入口、模型、配置和任务定义保留。Evidence 使用新的 `run_evidence.sh` / `run_evidence.py` 入口，训练与评估串行进行。

## 1. 放到当前 $HOME/CJJ

把完整 ZIP 放到 `$HOME/CJJ`，在远程主机执行：

```bash
cd "$HOME/CJJ"
unzip -n LS_Imagine_Clean_Evidence_20261020.zip
cd LS_Imagine_Clean_Evidence_20261020
```

脚本优先使用 `$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python`。若你的工作环境在别处，先设置 `export ENV_PYTHON=/绝对路径/bin/python`。本包不携带模型权重、MineDojo 安装和 Java，也不会替你安装或更新这些依赖。

资源优先从 `CORE_ASSETS`、当前项目、你现有的 `$HOME/CJJ/DreamerV3_Clean_PixelOnly_BinaryReward/CORE-CORE_DREAMERV3_MINEDOJO`、`$HOME/CJJ/CORE`、`$HOME/CJJ` 中寻找 `weights/mineclip_attn.pth`。同一个资源根目录还需包含原任务 YAML 指向的 `affordance_map/.../swin_unet_checkpoint.pth`。

资源位于其他位置时，在下面的命令末尾增加 `--assets-root /你的资源根目录`。需要的是包含 `weights` 和 `affordance_map` 的目录。

## 2. GPU1、seed1 从头训练

先检查已有 Python、Java8、CUDA、权重和本地 MineCLIP tokenizer：

```bash
bash run_evidence.sh --gpu 1 --seed 1 --check
```

`--check` 不启动 Minecraft。若 tokenizer 不在项目根目录，可先设置：

```bash
export MINECLIP_TOKENIZER=/你的已有目录/clip-vit-base-patch16
```

这里只使用已有本地 tokenizer，不下载文件；会验证词表与 CLIP 特殊 token，并为 MineCLIP 的相对路径查找建立链接。Java8 不在自动搜索位置时设置 `JAVA_HOME` 到你原来能运行的 JDK8。

开始正式实验：

```bash
bash run_evidence.sh \
  --gpu 1 --seed 1 \
  --task minedojo_harvest_log_in_plains \
  --steps 1000000 \
  --checkpoint-every 20000 \
  --logdir "$HOME/CJJ/LS_Imagine_Clean_Evidence_20261020/logdir/evidence_clean_seed1"
```

`--gpu 1` 在导入 PyTorch、MineCLIP 和 U-Net 前设置 `CUDA_VISIBLE_DEVICES=1`。训练模型、训练环境中的 teacher、评估模型和评估环境中的 MineCLIP 都在同一进程继承该设置。日志中的 `cuda:0` 是可见设备的逻辑编号，对应所选的物理 GPU1。评估期间暂停训练，两个环境实例在同一运行中持续复用。

若要断开 SSH 后继续运行，先在 `tmux new -s evidence_seed1` 中执行训练命令，运行后用 tmux 的前缀键再按 D 分离。重新连接后执行 `tmux attach -t evidence_seed1`。

## 3. 恢复同一实验

使用同一任务、seed、保存间隔、资源和日志目录，并加 `--resume`：

```bash
bash run_evidence.sh \
  --gpu 1 --seed 1 \
  --task minedojo_harvest_log_in_plains \
  --steps 1000000 \
  --checkpoint-every 20000 \
  --logdir "$HOME/CJJ/LS_Imagine_Clean_Evidence_20261020/logdir/evidence_clean_seed1" \
  --resume
```

这里的 `--steps` 是累计目标步数，不是再追加多少步。恢复读取最新完整 checkpoint 的真实步数、更新次数、评估调度、模型、优化器、replay 和 RNG。新建 Minecraft 环境会开始新 episode；不会恢复 Minecraft 中断时的物理世界和 policy carry。中断前尚未保存的动作需从最近 checkpoint 重新计算，日志追加并记录 `resume_boundary`，不会清空旧日志。

原 Clean 默认保存间隔为 100,000。本示例明确改为每 20,000 个真实训练动作保存；预填充 2,500 步计入累计步数，因此首次定期保存为 20,000，而不是 102,500。默认保留最新两份完整 checkpoint。每个 checkpoint 是含完整 replay 的压缩快照，随数据量增大，占用和保存时间也会增加；默认要求至少保留 8 GiB 空闲空间。

保存目录为 `checkpoints/step_0000020000` 等，`latest.json` 指向最近提交成功的快照。保存采用临时目录、校验和与原子提交；同一步数已存在时拒绝覆盖。已有实验忘记加 `--resume` 会报错；同一日志目录同时启动两个进程也会被拒绝。恢复时校验模型、环境源码、配置及权重身份，允许改变 GPU、资源所在路径和累计目标步数。

旧 WM 套件和原 LS 的 checkpoint 不能用本入口直接恢复。此移植版本应开始一个新的实验目录。

## 4. 评估保存的模型

```bash
bash run_evidence.sh \
  --gpu 1 --eval-only \
  --logdir "$HOME/CJJ/LS_Imagine_Clean_Evidence_20261020/logdir/evidence_clean_seed1" \
  --eval-episodes 100
```

默认读最新完整 checkpoint，也可加 `--checkpoint-step 20000` 指定仍保留的快照。训练中定期评估默认每 20,000 步、10 episodes；与原 Clean 一样，预填充后先评估初始化策略，再进行预训练。训练时的评估使用同一任务定义和 `actor.mode()`，不会自动改成 `test_` 任务。上面的独立评估也遵循该规则。

## 5. 环境与方法的边界

| 项目 | 本版本行为 |
|---|---|
| 任务与奖励 | 读取原 Clean `task_specs.yaml`；保留 native task reward，日志中的 return/success 不混入 shaping |
| 世界与重置 | 原 Clean SemifastResetWrapper；该 harvest 任务 fast_reset=5；不加入固定世界库、WorldPlan 或额外 simulator 重建 |
| 图像与动作 | 原 HUD 去除、64×64 RGB、12 个真实动作、sticky jump、pitch 限制和 repeat |
| 终止 | 原 is_first/is_last/is_terminal；transition discount 保留 Clean 的 `1-done` 口径 |
| 训练时评估 | 同任务、持续复用的独立评估环境、串行、actor.mode() |
| Evidence | RGB 编码、观察记忆、联合 S/Z 动力学、FC、RC、原生 imagined actor–critic |
| 语义 teacher | 仅训练生成当前视角 heatmap 标签；不作为 encoder/policy 输入；评估不加载 U-Net |
| 学习奖励 | native task reward + 原 Clean MineCLIP record-high intrinsic，两个 reward head 分开学习；RC 只检查真实 task reward |
| LS 专属行为 | 原 LS 入口保留；Evidence 不使用 LS concentration shaping、zoom、虚拟跳跃分支和标签尾段 |

MineCLIP 的 record-high intrinsic 沿用 Clean 的原实现，不能解释成相邻时刻分数差。两种方法共享环境协议，但算法、语义信息使用方式和辅助奖励机制存在方法本身的差异；本移植不构成性能优于 LS 的证据。

## 6. 已完成验证

57 项自动测试通过，覆盖真实 Clean wrapper 的 HUD、动作、奖励、终止、重置和默认 LS 分支；覆盖 Evidence 记忆、FC/RC 梯度、actor–critic 更新、标签不泄漏以及完整 checkpoint 恢复。模型测试运行实际小尺寸网络；环境测试保留真实 wrapper，替换底层 Minecraft 物理与 teacher。

测试环境为 CPU/Python 3.12、PyTorch 2.5.1、NumPy 1.26.4、Gym 0.21 原源码；新增代码另做 Python 3.9 语法检查。未在你的远程主机实际执行 Minecraft/Java/Gradle 启动、GPU 正式模型更新和百万步训练，因此这些运行结果仍需服务器验证。没有修改系统 Gradle 缓存或 MineDojo 安装。

`verification/` 中提供原始 ZIP 差异清单和验证结果。除 `envs/tasks/minedojo/__init__.py` 增加可选 Evidence 分支外，原包的任务、环境 wrapper、LS 模型、入口和配置逐字节保留。
