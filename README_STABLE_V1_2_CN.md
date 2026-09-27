# Stable V1.2（克制版）

Stable V1.2 直接继承 Stable V1.1，只加两个保护层，不改网络结构、不改 Full Heatmap、不改 S-Aff、不改 S5A、不改 Outcome gate，也不改 Actor/Critic 超参数。

## 1. Proto-only Gradient Guard

Stable V1.1 已经用 teacher reliability gate 控制 Proto 的“证据质量”。V1.2 再检查 Proto 对 **S prior dynamics** 的梯度是否与核心 world-model objective 冲突。

核心 objective 包含原有重建/KL、S-Aff 和 S5A；Outcome 保持 V1.1 的 evidence gate，不纳入这个 guard。

每 16 次 model update 检查一次：

- 梯度方向一致：Proto gate = 1；
- 梯度轻微冲突：平滑降低；
- cosine <= -0.20：conflict gate 最低 0.25；
- Proto 对 S 的加权梯度范数若超过 core gradient 的 0.35 倍，再做 norm cap；
- 总 gate 保留 0.10 下限；
- gate 用 EMA=0.80 平滑，避免单个 batch 触发骤变。

它不是完整 PCGrad：**不改写梯度向量，只缩小 Proto scalar loss**，因此改动和计算开销都更小。

实际 Proto 权重为：

`0.01 × reliability_gate × gradient_guard_gate`

新增指标：

- `health/proto_grad_cos`
- `health/proto_grad_norm_ratio`
- `health/proto_grad_gate`
- `health/proto_grad_checked`

## 2. 10% Success-aware Replay

不建立新的 replay buffer，也不修改环境数据。只在原来的 episode cache 上做一个 90/10 mixture：

- 90%：原来的 length-proportional replay；
- 10%：从“曾出现真实正 reward”的 episode 中抽样。

没有成功 episode 时，自动退化为原 sampler。

窗口本身不要求包含最终 reward，这样也会保留成功前的 near-success context。

新增指标：

- `health/replay_success_source`：当前 batch 有多少 sequence 使用了 success-priority 路径；
- `health/replay_positive_sequence`：当前 batch 中实际含正 reward 的 sequence 比例；
- `health/replay_positive_step`：正 reward timestep 比例。

## 启动

```bash
python audit_stable_v1_2.py
python test_stable_v1_2.py
bash ./scripts/train_stable_v1_2.sh harvest_log_in_plains 0
bash ./scripts/train_stable_v1_2.sh shear_sheep 0
```

日志目录：`./logdir/stable_v1_2/`

## 建议第一轮不要再调参数

优先观察：

- Proto reliability gate 与 gradient guard gate 是否只在弱/冲突阶段下降；
- success replay 实际比例是否接近 0.10；
- Sheep 的 `ImagRewardStd -> ValueStd -> ActorGrad` 是否不再长期坍缩；
- Tree seed2 这种 bad bootstrap 是否改善，而 seed0/1 的原有强性能是否保留。
