# Generic Task-Evidence Prototype Utility v2

本版本把上一版按 `time-to-success` 划分的 Prototype Utility 改成了 **跨任务通用的 task-evidence 原型构造**。

## 为什么修改

旧版默认：

```text
Progress -> Near -> Ready -> Success
```

类别由“距离真实 reward 还有多少步”决定。这种方法对不同 MineDojo 任务不稳健：剪羊、砍树、挖矿、收集等任务的完成时间尺度完全不同，batch_len=32 也限制了时间标签最多只能向前覆盖一个 replay chunk。

v2 不再为某一个任务人工定义阶段，也不使用 sheep-specific 的 `Search/Acquired/Ready` 规则。

## 通用四类

```text
0 = Low task evidence
1 = Medium task evidence
2 = High task evidence
3 = Real Success
-1 = Unknown / ambiguous
```

前三类来自每个任务都已有的 task-conditioned teacher signal：

1. **MineCLIP raw task score**：当前观察和该任务文本的语义匹配；
2. **Affordance heatmap top-k response**：任务相关视觉区域是否出现强响应，不要求目标位于画面中心；
3. **Intrinsic progress event**：现有 ClipWrapper 中“刷新历史最好 MineCLIP score”的事件，只给很小权重。

真实 environment reward 始终覆盖 pseudo label，作为 class 3 的硬 Success anchor。

## 为什么可以跨多个任务

三个 teacher 都由当前任务 prompt 条件化，而代码没有：

- sheep 名称；
- tree 名称；
- ore 名称；
- 固定交互距离；
- 固定成功步数。

而且 semantic / affordance / intrinsic 三个分量会在**当前 replay batch 内进行稳健 q05-q95 归一化**：

```text
q05 -> 0
q95 -> 1
```

所以不要求不同任务的 MineCLIP 或 heatmap 原始数值具有相同分布。

组合证据默认：

```text
0.65 * semantic
+ 0.30 * affordance
+ 0.05 * intrinsic-progress
```

随后根据组合证据的 batch 分位数自适应形成：

```text
Low:    lower evidence region
Mid:    middle evidence region
High:   upper evidence region
Success: real environment reward
```

阈值边界附近保留为 `Unknown=-1`，避免 noisy pseudo labels。

## 对 S 的训练路径

真实 posterior S：

```text
posterior S
 -> existing FixedRandomProjector (2560 -> 512)
 -> detached support
 -> EMA prototype bank
```

S5A imagined future：

```text
S_t + real action prefix
 -> img_step_s
 -> predicted future S
 -> same FixedRandomProjector
 -> cosine similarity to prototypes
 -> ProtoNet cross entropy
 -> gradient to predicted S / S dynamics
```

Prototype teacher 不直接反传到 posterior encoder。

## RGB-only inference 没变

新增 `task_score` 只写入 replay 作为 training-time auxiliary teacher。

当前 encoder 仍然：

```yaml
cnn_keys: '^image$'
mlp_keys: '$^'
```

因此 inference / evaluation 时 actor 仍然只依赖 RGB 经过世界模型得到的 `[S,Z]`，不会读取 heatmap、MineCLIP task score 或 prototype label。

## 不加入 actor reward

v2 仍然没有：

```python
imag_reward += prototype_utility
```

所以这是 representation supervision，不是 reward shaping。

## 默认参数

见 `configs.yaml`：

```yaml
label_mode: task_evidence
semantic_weight: 0.65
affordance_weight: 0.30
intrinsic_weight: 0.05
heatmap_topk_fraction: 0.05
robust_low_quantile: 0.05
robust_high_quantile: 0.95
stage_low_quantile: 0.30
stage_high_quantile: 0.70
boundary_margin: 0.05
loss_scale: 0.01
```

旧时间分层仍保留为消融：

```yaml
label_mode: temporal_success
```

但不是默认方法。

## 关键日志

```text
s_proto_support_low
s_proto_support_mid
s_proto_support_high
s_proto_support_success

s_proto_teacher_semantic_active
s_proto_teacher_affordance_active
s_proto_teacher_intrinsic_active

s_proto_evidence_mean
s_proto_evidence_std
s_proto_stage_low_threshold
s_proto_stage_high_threshold

s_proto_label_low_fraction
s_proto_label_mid_fraction
s_proto_label_high_fraction
s_proto_label_success_fraction
s_proto_label_unknown_fraction

s_proto_acc_h1
s_proto_acc_h2
s_proto_acc_h4
s_proto_acc_h8
s_proto_acc_h15
```

## 推荐实验

同一套代码直接分别运行多个任务，不为不同任务修改 Prototype 参数：

```text
harvest_log_in_plains
shear_sheep
mine_iron_ore
harvest_wool_with_shears
...
```

首先比较：

```text
Outcome
vs.
Outcome + Generic Task-Evidence ProtoNet
```

如果多个不同任务都改善，才能支持“通用 sparse-utility grounding”这一方法主张。
