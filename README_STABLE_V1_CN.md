# Stable V1

Stable V1 以当前历史最强版本为锚点，不改 Dual S/Z、Full Heatmap、S-only Affordance、S5A、Proto、Outcome 的主体结构。
目标不是继续加模块，而是降低稀疏任务证据引起的 bootstrap 方差。

## 保持不变

- Full latent heatmap reconstruction：保持原版开启、原权重。
- S-only Affordance：`affordance_s_scale: 1.0`。
- S5A：horizons `[1,2,4,8,15]`、weights `[1,1,.75,.5,.25]`、scale `0.05`。
- Generic Proto：总 loss scale 仍为 `0.01`。
- Balanced Outcome：总 loss scale 仍为 `0.01`。
- Inverse 与 Z adversary 继续关闭。

## Stable V1 修改

### 1. Proto EMA quantile teacher

原版每个 replay batch 独立计算 q05/q95 归一化边界和 q30/q70 stage 边界，因此 replay distribution 改变时 Low/Mid/High 的含义会快速漂移。

Stable V1 使用：

- `quantile_mode: ema`
- `quantile_ema: 0.99`
- `boundary_margin: 0.10`

batch quantile 仍然记录，但真正用于归一化/分层的是 EMA 边界。

### 2. 更慢的 prototype bank

`prototype_ema: 0.95 -> 0.98`，减少 prototype 随 replay 快速移动。

### 3. Success prototype cold-start protection

`success_min_support: 256`。

Success prototype 可以累计 support，但在累计 exposure 不足 256 前不进入 Proto 分类 logits，也不会作为 future-S 的有效 Success target。
这避免少量偶然成功样本立即定义高权重 Success prototype。

### 4. Outcome evidence gate

Outcome head 的 dense posterior calibration 始终训练；只有会向 predicted S dynamics 反传的 rollout Outcome loss 被 gate。

默认：

- `evidence_gate_ema: 0.99`
- `evidence_gate_low: 5e-5`
- `evidence_gate_high: 5e-4`
- `evidence_gate_horizon: 15`

有效 gate：

`gate = clip((positive_ema - low) / (high - low), 0, 1)`

因此真实 positive return evidence 极少时，Outcome head 先校准，不急着塑形 S；证据积累后再逐步恢复完整 rollout Outcome 梯度。

## 新增直观指标

Stable V1 会继续把所有原始指标写入 `metrics.jsonl`，同时额外写入：

- `health/s5a_cos15`
- `health/cf_margin15`
- `health/response_gap15`
- `health/reanchor_gain15`
- `health/proto_acc15`
- `health/proto_known`
- `health/proto_success_ready`
- `health/proto_success_support_ratio`
- `health/proto_semantic_span`
- `health/proto_affordance_span`
- `health/outcome_gate`
- `health/outcome_positive_ema`
- `health/outcome_corr15`
- `health/imag_reward_std`
- `health/value_std`
- `health/actor_grad_norm`
- `health/policy_collapse_alert`

`policy_collapse_alert=1` 仅作为诊断警报，不参与训练。默认条件为：

- `value_std < 0.01`
- `actor_grad_norm < 0.003`
- `imag_reward_std < 0.001`

三者同时满足。

控制台 Stable V1 模式会输出两行简要训练摘要，例如：

```
[TRAIN 202500] model=... image=... heat=... S-Aff=... fps=...
[HEALTH] S5A(cos/margin/gap)=.../.../... | Proto(acc/known/succReady)=.../.../... | Outcome(gate/posEMA/corr)=.../.../... | Policy(Rstd/Vstd/Agrad)=.../.../... | ALERT=0
```

所有完整原始指标仍保存在 JSONL 和 TensorBoard 中，没有删减。

## 启动

```bash
bash ./scripts/train_stable_v1.sh harvest_log_in_plains 0
```

剪羊：

```bash
bash ./scripts/train_stable_v1.sh shear_sheep 0
```

原始最强 baseline 仍可继续用原脚本：

```bash
bash ./scripts/train.sh harvest_log_in_plains
```

这样 baseline 和 Stable V1 可以在同一代码包内严格对照。
