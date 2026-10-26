# 新组合消融、固定回放诊断与配对种子（2026-10-25）

本版基于 `LS_Imagine_Evidence_Ablations_20261021`。新增 `no_memory_no_rc`、`no_memory_no_fc_rc`，并让 LS/no_memory 使用同一启动入口和明确相同的训练、评估、保存预算。没有调整 FC/RC 权重、奖励定义或 Clean 世界重置协议。

## 1. 安装到新的目录

保留正在运行的旧目录。首次安装：

```bash
cd "$HOME/CJJ"
unzip -n LS_Imagine_Evidence_Ablations_20261025.zip
cd LS_Imagine_Evidence_Ablations_20261025
conda activate ls_imagine
```

如果 `LS_Imagine_Evidence_Ablations_20261025` 已经用于实验，不要混入另一份源码；换一个空的安装位置。脚本优先使用 `$HOME/CJJ/miniconda3/envs/ls_imagine/bin/python`，不同安装位置用 `export ENV_PYTHON=/实际环境/bin/python` 指定。

已有资源沿用原位置，例如：

```bash
export CORE_ASSETS="$HOME/CJJ/LS_Imagine_Clean_Evidence_20261020"
bash run_experiment.sh --method evidence_dreamer --ablation no_memory_no_rc --gpu 1 --seed 1 --check
bash run_experiment.sh --method ls_imagine --gpu 1 --seed 1 --check
```

`CORE_ASSETS` 必须实际包含 MineCLIP 与任务 U-Net 权重。若资源在别处，使用真实路径。不要下载或重建原本可以工作的 Java/Gradle 环境。`--check` 不启动 Minecraft；完整环境重置仍需在远程执行一次独立 `--stage smoke` 实验验证。

`--gpu 1` 在导入 PyTorch 前选择物理 GPU1。进程内训练、评估和 teacher 的 `cuda:0` 是唯一可见设备的逻辑编号，不表示偷偷切回物理 GPU0。

## 2. 新增两组组合对照

| 消融名称 | Memory | FC | RC | 语义监督、task/clip 奖励头 |
|---|---|---|---|---|
| `no_memory` | 关 | 开，0.05 | 开，0.01 | 保留 |
| `no_memory_no_rc` | 关 | 开，0.05 | 关 | 保留 |
| `no_memory_no_fc_rc` | 关 | 关 | 关 | 保留 |

这三组分别回答“无记忆时 RC 是否有收益”和“无记忆、无 RC 时 FC 是否有收益”。`no_memory_no_fc_rc` 仍保留 S/Z 动力学与局部/后验热图监督，因此不能把它称作官方 DreamerV3。

单独启动 GPU1、seed1：

```bash
bash run_experiment.sh --method evidence_dreamer --ablation no_memory_no_rc \
  --gpu 1 --seed 1 --steps 1000000

bash run_experiment.sh --method evidence_dreamer --ablation no_memory_no_fc_rc \
  --gpu 1 --seed 1 --steps 1000000
```

两条命令在同一终端依次执行，前一个结束后才执行后一个。默认分别写入新代码目录的 `logdir/evidence_clean/<ablation>/formal/minedojo_harvest_log_in_plains/seed_1`。查看准确参数可在命令最后加 `--dry-run`，它不会启动训练。

## 3. LS 与 no_memory 配对种子

默认 seeds 为 `0 1 2`，方法为 `ls_imagine no_memory`，共六个任务。每个任务都是独立串行实验。先打印计划：

```bash
bash run_paired_benchmark.sh --gpu 1 --seeds 0 1 2
```

实际在 GPU1 上依次跑六个任务：

```bash
bash run_paired_benchmark.sh --gpu 1 --seeds 0 1 2 --execute
```

默认顺序为 seed0 的 LS/no_memory，然后 seed1、seed2。任一任务失败即停止队列。默认正式预算统一为 1,000,000 个训练环境动作，每 20,000 步评估 10 回合、保存 checkpoint。LS 计数仍包含其原算法的长期标签尾段动作。模型推理/更新次数和每步计算成本仍可能不同。

同一个 `--experiment-root` 下的目录结构：

```text
logdir/paired_20261025/<方法或消融名>/formal/minedojo_harvest_log_in_plains/seed_<seed>/
```

该位置与单次启动器的默认目录不同，防止混入历史实验。可用 `--experiment-root /新的绝对路径` 指定另一套实验。

如果 GPU0、GPU1 都空闲，可在两个 tmux 会话里分别运行：

```bash
# 会话一：GPU0 上依次跑三个 LS 种子
bash run_paired_benchmark.sh --methods ls_imagine --gpu 0 --seeds 0 1 2 --execute
```

```bash
# 会话二：GPU1 上依次跑三个 no_memory 种子
bash run_paired_benchmark.sh --methods no_memory --gpu 1 --seeds 0 1 2 --execute
```

两条命令使用同一实验根目录，但方法子目录互不相同；不要同时提交相同的“方法 + seed”两次。如果 GPU0 有其他任务，只使用空闲 GPU 的串行方案。

新增两组也可批量运行：

```bash
bash run_paired_benchmark.sh \
  --methods no_memory_no_rc no_memory_no_fc_rc \
  --gpu 1 --seeds 0 1 2 --execute
```

“配对 seed”仅保证共同的实验种子、预算与 Clean 环境配置，不保证逐回合世界一一对应，也不保证两个不同算法的参数初始化一致。本版没有注入固定世界库，没有改变 SemifastResetWrapper。

## 4. 恢复必须对应准确的目录和版本

通过单次启动器默认目录启动的组合实验：

```bash
bash run_experiment.sh --method evidence_dreamer --ablation no_memory_no_rc \
  --gpu 1 --seed 1 --steps 1000000 --resume
```

通过配对启动器启动的 no_memory seed1，恢复同一个配对目录：

```bash
bash run_paired_benchmark.sh --methods no_memory --gpu 1 --seeds 1 --resume --execute
```

恢复配对目录的 LS seed1：

```bash
bash run_paired_benchmark.sh --methods ls_imagine --gpu 0 --seeds 1 --resume --execute
```

如果首次用了自定义 `--experiment-root`、资源路径或其他训练参数，恢复时沿用它们。三个只读诊断运行选项 `diagnostics.enabled/every/gradients` 允许在同一版本的 resume 时改变，例如关闭梯度诊断或降低频率；已捕获 bank 保持原样，调整间隔后从恢复步数重新计算下一次诊断时点。`diagnostics.batch_size` 仍是受保护的实验配置，不能借恢复换掉固定样本。未显式覆盖时，启动器使用本版默认诊断选项，因此希望保持自定义频率时仍应重复传入它。

`--steps` 是累计目标，不是追加步数。配对启动器的 `--resume` 要求每个选中的任务都已有 checkpoint；混合“已启动”和“未启动”任务时，分别筛选 methods/seeds 后执行，不会自动把缺失任务从零开始。

**不要使用本包直接 resume 旧 20261021 实验。** 新源码 hash 与旧 checkpoint 的身份不同，会被拒绝。旧实验要继续使用旧 20261021 代码、原命令和原目录。新消融和诊断实验使用新目录；不要修改 checkpoint、删除 `provenance.json` 或绕过 guard。相同版本内会恢复学习器、优化器、replay、RNG、调度和固定诊断 bank，Minecraft 物理场景仍从新 episode 开始。

## 5. 诊断默认配置

| 配置键 | formal 默认 | 作用 |
|---|---:|---|
| `diagnostics.enabled` | `true` | 开启只读诊断 |
| `diagnostics.every` | `20000` | 诊断间隔，按累计训练动作计数；smoke 为 2500 |
| `diagnostics.batch_size` | `2` | 每个固定 bank 最多两条回放序列 |
| `diagnostics.gradients` | `true` | 测量 base/FC/RC 在相同动力学参数上的梯度 |

所有结果追加到原 `metrics.jsonl`，不新增另一套训练日志。每次包含 `diagnostics_start` 和 `diagnostics`；失败时显式记录 `diagnostics_failed`，保留错误信息并继续训练，不把失败写成零误差。`seconds` 记录诊断墙钟成本，`checkpoint` 事件另有 `checkpoint_seconds` 记录保存成本。

诊断不调用 optimizer.step，不修改损失权重，不新增训练样本或环境交互。它用独立、可恢复的随机流进行前向计算；`autograd.grad` 只测量梯度，原参数 `.grad`、优化器状态与训练随机流保持不变。它有额外计算和显存成本，“只读”不等于免费。

如需在新实验中降低诊断开销，可选择只测前向：

```bash
bash run_experiment.sh --method evidence_dreamer --ablation no_memory_no_rc \
  --gpu 1 --seed 2 --set diagnostics.gradients=false
```

配对计划使用对应选项：

```bash
bash run_paired_benchmark.sh --gpu 1 --seeds 0 1 2 \
  --diagnostics-every 40000 --diagnostics-batch-size 1 --no-diagnostic-gradients
```

最后一行仍只打印计划，加 `--execute` 才会启动。`--no-diagnostics` 可完全关闭 Evidence 诊断；这些覆盖项只传给 Evidence，不改变 LS。正式配对实验应让各 Evidence seed 使用同样的诊断设置。

## 6. 两个固定 bank 的含义

`event=diagnostics` 的 `banks` 内可能出现：

| bank | 固定样本来源 | 解释限制 |
|---|---|---|
| `natural` | 首次可用时，从本 run 已记录的 replay 中按转换数量加权抽样 | 用于观察同一批真实数据上的指标变化；不是独立测试集 |
| `reward_nonzero` | 首次出现合适非零任务奖励后，抽取包含奖励前动作前缀的序列 | 特意富集非零奖励，不代表自然成功率或奖励发生频率 |

每个 bank 一经捕获即冻结，保存 `fingerprint`、`captured_at_step`、`locations`，并写入 checkpoint。后续诊断和 resume 重用完全相同的数组。两个 bank 可能在不同训练步数才形成；早期没有奖励时，`pending_banks` 会列出 `reward_nonzero`，不能把它当“非零误差为 0”。

这是**每个 run 内部**的固定探针，不是不同 run 自动共享的数据，也不是 held-out benchmark。固定输入能减少训练分布变化带来的混淆，但 latent 后验仍随模型变化；跨方法/种子比较时仍要结合真实评估，不能把这些损失直接当最终排名。

## 7. 主要诊断字段

下表字段位于 `banks.<bank名>.metrics`。`H` 为 `h1/h2/h4/h8/h15` 等实际配置 horizon；smoke 的 horizon 不同。

| 字段 | 含义 |
|---|---|
| `rc_H_nonzero_prediction_mean` / `target_mean` | 非零真实回报片段上的预测均值和目标均值 |
| `rc_H_nonzero_bias` | 预测减目标，正值表示该批片段平均高估，负值平均低估 |
| `rc_H_nonzero_mae` / `rmse` | 非零片段的绝对误差/均方根误差 |
| `rc_H_zero_*`、`rc_H_all_*` | 同样统计分别作用于零目标、全部目标 |
| `rc_all_horizons_*` | 按配置中的 horizon 权重汇总；不是另一个 rollout horizon |
| `*_prefixes` | 合法动作前缀样本计数，不是独立成功 episode 数 |
| `fc_loss` / `rc_loss` | 固定 bank 上未乘训练权重的辅助目标值 |
| `fc_weighted_loss` / `rc_weighted_loss` | 乘当前训练配置权重后的值 |
| `fc_categorical` / `fc_hidden` / `fc_heat` | FC 分项，配有分子分母；`fc_heat_available` 指示是否具备热图头和标签 |
| `grad_base_norm` | 基础世界模型目标在共同动力学参数集合上的梯度范数 |
| `grad_fc_raw_norm` / `grad_rc_raw_norm` | 未乘辅助损失权重的梯度范数 |
| `grad_fc_weighted_norm` / `grad_rc_weighted_norm` | 按当前权重缩放后的梯度范数 |
| `grad_fc_base_cosine` / `grad_rc_base_cosine` | 辅助目标与 base 梯度方向的余弦；不是损失数值比 |

关闭 RC/FC 时，诊断仍可测量对应的未加权误差及假设启用时的原始梯度，但它们的**训练贡献仍为零**。不要因为 `no_rc` 的诊断中出现 `rc_loss` 就认为 RC 被重新打开。原 `event=train` 的 loss 仍按实际训练开关计算。

空样本统计、零梯度导致无定义的余弦等使用 `null`，同时保留 `*_numerator`、`*_denominator`。跨记录汇总比值应先累加分子与分母，再相除；不能把 `null` 当零后求平均。关闭 `diagnostics.gradients` 时不会产生这批梯度指标。

Memory 使用两个前缀，分别测量真实观察后验构建路径 `memory_posterior_*` 与记录动作 rollout 路径 `memory_recorded_*`：

| 后缀 | 含义 |
|---|---|
| `enabled`、`read_mode_code` | 是否启用；读取模式 0=none，1=start_only，2=each_step |
| `available_fraction`、`active_read_fraction` | 有历史记忆的转换比例、实际执行读取的转换比例 |
| `gate_mean`、`gate_low_fraction`、`gate_high_fraction` | 门控均值及小于 0.05 / 大于 0.95 的比例 |
| `attention_null_mass`、`attention_real_mass` | 注意力分给空 token 与真实记忆的质量 |
| `attention_entropy_normalized` | 按有效 key 数归一化的注意力熵，均匀读取约为 1 |
| `attended_age_steps` | 按真实记忆注意力加权的年龄，不把空 token 记作新观察 |
| `gated_delta_to_state_norm` | 门控读取残差范数相对于读取前状态范数 |
| `effective_delta_to_state_norm` | 完成读取路径后状态实际变化范数的相对大小 |

Memory 关闭、缓存为空或该步按模式跳过读取时，没有实际读取样本的门控/注意力统计为 `null`，不表示“错误为零”。注意力统计使用 detached Q/K 另行测量，不替换原模型前向读取结果。

## 8. 只读汇总配对结果

新工具只读取 `metrics.jsonl`，结果打印到终端，不修改日志，不启动训练：

```bash
python summarize_paired_benchmark.py --seeds 0 1 2 --last-evals 10
```

默认取所有完整配对 run 共同拥有的最新评估 checkpoint，向前取 10 个共同评估点。指定步数或只看单个 checkpoint：

```bash
python summarize_paired_benchmark.py --seeds 0 1 2 --step 502500 --last-evals 10
python summarize_paired_benchmark.py --seeds 0 1 2 --step 502500 --last-evals 1
```

比较无记忆三组时：

```bash
python summarize_paired_benchmark.py \
  --methods no_memory no_memory_no_rc no_memory_no_fc_rc \
  --reference no_memory --seeds 0 1 2 --last-evals 5
```

工具输出每 seed 成功数/评估回合数、各方法跨 seed 等权均值和样本标准差、相对 reference 的逐 seed 差值（百分点）、实际纳入种子数和缺失任务。只有一个 seed 时标准差为 `null`。缺少另一方法结果的 seed 不进入配对统计；没有足够共同评估点会明确报错。

LS 原有 flat `eval_success` 只计一次，嵌套在 `result` 中的运行摘要不重复计数。同一步有多次 flat 评估时采用最后一条并报告重复步数。遇到 `resume` 或 `resume_boundary` 时，会撤销恢复步数之后的旧评估记录，并在 `evaluations_invalidated_by_resume` 报告，避免把 checkpoint 之后尚未重跑的数据当成当前最新结果。日志末行未写完会报出行号，不悄悄忽略；源日志不会被改写。

“最近 5/10 轮”是在多个 checkpoint 之间汇总，不能写成“最终固定模型测试了 50/100 回合”。同 seed 比较也不是同一批世界上的严格配对检验。本工具不生成显著性结论。

## 9. 验证范围

包内测试覆盖小模型真实更新、固定 bank 保持、诊断随机流/参数/优化器不变、消融开关、配对 CLI 与恢复保护。它们不替代远程 Minecraft/Java/Gradle 和正式 GPU 资源验证。新实验的实际耗时应结合 `seconds`、`checkpoint_seconds`、评估环境动作数和设备信息分析，不能把“成功更早结束评估”全部当作模型计算提速。
