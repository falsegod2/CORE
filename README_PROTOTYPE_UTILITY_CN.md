> **注意：此文件描述的是 v1 temporal-success 原型。当前默认版本已升级为跨任务的 Generic Task-Evidence Prototype Utility v2。请优先阅读 `README_GENERIC_PROTOTYPE_UTILITY_CN.md`。**

# ISO3-SAff-SOnly5A-NoInverse-BalancedOutcome-ProtoNet-Clean

这是在用户上传的：

`ISO3-SAff-SOnly5A-NoInverse-BalancedOutcome-Clean.zip`

基础上直接修改得到的完整工程版本。

## 保留的原方法

原代码以下核心路径保持不变：

- Dual S/Z RSSM
- S branch action-conditioned prior
- Z branch action-unconditioned prior
- RGB-only encoder
- Full `[S,Z] -> RGB + heatmap`
- S-only affordance supervision (S-Aff)
- S-only multi-step action-conditioned consistency (S5A)
- FixedRandomProjector `S: 2560 -> 512`
- Balanced Multi-Step Outcome
- Dense posterior Outcome calibration
- Counterfactual action diagnostics
- Re-anchor diagnostics
- Actor/Critic 仍使用完整 `[S,Z]`
- MineCLIP intrinsic protocol 保持原样
- Inverse loss 仍为 0
- Z adversary loss 仍为 0

## 新增：Few-Shot Prototypical Utility

目标是解决 `shear_sheep` 中：

- S-Aff 已能把 task relevance 压入 S；
- S5A 已能形成很强的 action-consequence sensitivity；
- 但真实 environment reward 过于稀疏，使 Balanced Outcome 几乎得不到可靠 utility supervision。

新模块把少量真实成功轨迹扩展为四个 ordinal task stage：

```text
Progress -> Near -> Ready -> Success
```

其中：

- Success：当前状态收到真实 environment success reward；
- Ready：未来 1~4 步内出现 success；
- Near：未来 5~15 步内出现 success；
- Progress：更早、但在当前 sampled episode/chunk 内确实能观察到未来 success；
- Unknown：当前 chunk 内看不到未来 success。

### 关键设计

**reward == 0 不自动等于 negative。**

无成功证据的状态保持 `Unknown(-1)`，不进入 prototype classification loss。

## 数据流

```text
real posterior S
      |
      | no_grad
      v
existing FixedRandomProjector (2560 -> 512)
      |
      v
EMA task-stage prototypes
 Progress / Near / Ready / Success


S_t + real action prefix
      |
      v
existing S5A img_step_s rollout
      |
      v
predicted future S_hat
      |
      v
same FixedRandomProjector
      |
      v
ProtoNet cosine classification loss
      |
      v
gradient -> predicted future S -> existing S dynamics
```

Prototype bank 本身没有 trainable MLP。

## Prototype 不进入 Actor reward

v1 坚持：

```text
Prototype Utility = representation supervision
```

而不是：

```text
Prototype Utility = intrinsic reward
```

因此 `ImagBehavior` 没有加入任何 prototype reward。

## 总 loss

原：

```text
L_total =
    L_base
  + L_S-Aff
  + 0.05 L_S5A
  + 0.01 L_Outcome
```

现在：

```text
L_total =
    L_base
  + L_S-Aff
  + 0.05 L_S5A
  + 0.01 L_Outcome
  + 0.01 L_Proto
```

## 新增日志

总指标：

```text
s_proto_enabled
s_proto_loss
s_prototype_scale
s_prototype_weighted_loss

s_proto_initialized
s_proto_known_fraction
s_proto_unknown_fraction

s_proto_support_progress
s_proto_support_near
s_proto_support_ready
s_proto_support_success

s_proto_batch_progress_count
s_proto_batch_near_count
s_proto_batch_ready_count
s_proto_batch_success_count
```

多 horizon：

```text
s_proto_loss_h1
s_proto_loss_h2
s_proto_loss_h4
s_proto_loss_h8
s_proto_loss_h15

s_proto_acc_h1
s_proto_acc_h2
s_proto_acc_h4
s_proto_acc_h8
s_proto_acc_h15

s_proto_num_h1
s_proto_num_h2
s_proto_num_h4
s_proto_num_h8
s_proto_num_h15
```

## 配置

`configs.yaml` 的 `s_multi_step_consistency` 下新增：

```yaml
prototype_utility:
  enabled: true
  loss_scale: 0.01
  temperature: 0.10
  prototype_ema: 0.95
  ready_steps: 4
  near_steps: 15
  progress_steps: null
  positive_threshold: 1.0e-6
```

## 修改文件

### 1. `s_prototype_utility.py`（新增）

实现：

- `steps_to_next_success`
- `build_progress_labels`
- `PrototypeUtilityBank`

### 2. `s_multistep_consistency.py`

新增：

- Prototype config arguments
- EMA prototype bank
- posterior-S support update
- S5A predicted future-S query loss
- 多 horizon ProtoNet loss/accuracy/count 指标
- forward 新增 `prototype_total` 返回值

没有改变原 S5A rollout 的 action indexing。

### 3. `models.py`

新增：

- 读取 `prototype_utility` config
- `_s_prototype_enabled`
- `_s_prototype_scale`
- 接收 `s_prototype_loss`
- `+ self._s_prototype_scale * s_prototype_loss`
- 记录 prototype weighted loss

没有修改 `ImagBehavior` reward。

### 4. `configs.yaml`

开启 Prototype Utility，默认 scale = 0.01。

### 5. `test_prototype_utility.py`（新增）

验证：

- success 向前传播成四阶段 label；
- 无 success trajectory 保持 Unknown；
- 开启 prototype 不改变 S5A 随机数消费；
- S5A 原 loss 不变；
- Prototype loss 的梯度能到现有 S transition；
- Prototype metrics 正常产生。

### 6. 原测试适配

`SOnlyMultiStepRSSMConsistency.forward()` 现在返回：

```python
s5a_loss, outcome_loss, prototype_loss, metrics
```

因此更新：

- `test_s_outcome.py`
- `test_s5a_diagnostics_invariance.py`

### 7. 审计

新增：

`audit_prototype_utility.py`

并把旧的 `audit_saff_inverse_sonly5a.py` 中已经过时的
`inverse_loss_scale: 1.0` 期望改为当前 NoInverse build 的 `0.0`。

## 推荐第一条实验

直接跑：

```text
shear_sheep seed0
```

和原始：

```text
BalancedOutcome-Clean sheep_seed0
```

对比。

第一阶段最重要的不是马上看 success，而是先看：

1. `s_proto_initialized` 是否能达到 3~4；
2. `s_proto_support_success` 是否开始积累；
3. `s_proto_acc_h1/h4/h8/h15` 是否逐渐高于随机水平；
4. `s_proto_num_h*` 是否有足够有效 query；
5. 最后再看 500K~800K eval 是否比原 Outcome 更容易 bootstrap。

如果 prototype 长期只有 1 个 class 被初始化，则 ProtoNet loss 会安全退化为 0，
不会在没有足够 few-shot support 的情况下强行训练。
