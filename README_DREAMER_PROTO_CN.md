# Dreamer + Generic Proto（单流 Full-Latent 消融）

这个 profile 专门回答一个问题：

> **Generic Prototype 本身是否有效，还是它只有放在 Dual S/Z 的 controllable S 上才有效？**

## 1. 这不是 Dual S/Z

运行：

```bash
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer_proto 0
```

实际结构是：

```text
RGB
 ↓
Dreamer encoder
 ↓
单一 RSSM: (stoch, deter)
 ↓
full latent x_t
```

没有：

- Dual S/Z；
- S-Aff；
- S5A consistency loss；
- Balanced Outcome；
- inverse loss；
- Z adversary。

## 2. Generic Proto 放在哪里

Dual 版本原来是：

```text
S_t + replay action prefix
    ↓ S prior transition
S_hat_{t+k}
    ↓ fixed random projector 512D
Generic Proto CE
```

本消融改成：

```text
Dreamer full latent x_t + replay action prefix
    ↓ native Dreamer RSSM img_step
x_hat_{t+k}
    ↓ fixed random projector 512D
Generic Proto CE
```

因此唯一关键变化是：

```text
S branch  →  ordinary Dreamer full latent
```

Generic Proto 的 teacher、prototype bank、horizons、weights、temperature、EMA、projection seed 均保持一致。

## 3. Teacher 与 GenericTask 完全相同

默认：

```yaml
semantic_weight: 0.65
affordance_weight: 0.30
intrinsic_weight: 0.05
```

形成：

```text
Low / Mid / High / real Success
```

其中 Success 仍只由真实环境 reward 硬锚定。

Heatmap 在这个实验中**只作为 Proto teacher**。因为：

```yaml
module_saff: false
decoder.cnn_keys: '^image$'
```

所以不存在 latent -> heatmap reconstruction，也不存在隐藏 S-Aff loss。

## 4. 没有偷偷加入 S5A

本 profile：

```yaml
module_s5a: false
s_multi_step_consistency.enabled: false
s_multi_step_consistency.loss_scale: 0.0
```

Proto query 的未来 latent 使用 Dreamer 自己本来就有的：

```python
dynamics.img_step(...)
```

这只是 standard Dreamer latent dynamics rollout。

**不会计算**：

```text
1 - cosine(predicted future, posterior future)
```

因此它不是 `Dreamer + S5A + Proto`。

## 5. 运行命令

### Tree seed0

```bash
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer_proto 0
```

### Sheep seed0

```bash
bash ./scripts/train_ablation.sh shear_sheep ab_dreamer_proto 0
```

### seed1

把最后一个参数改为：

```bash
1
```

## 6. 建议一起跑的 2×2 消融

```text
A Dreamer
B Dreamer + Generic Proto(full latent)
C Dreamer + Dual S/Z
D Dreamer + Dual S/Z + Generic Proto(S)
```

命令：

```bash
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer_proto 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dual 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dual_proto 0
```

这样可分别计算：

```text
Proto main effect = B - A
Dual main effect  = C - A
Interaction       = D - C - B + A
```

如果 Interaction 明显 > 0，就支持：

> Generic Proto 放在 controllable S 上，比直接放在普通 Dreamer latent 上更有效。

## 7. 可调参数

```yaml
proto_scale: 0.01
proto_latent_source: full
proto_label_mode: task_evidence
```

以及 `s_multi_step_consistency.prototype_utility` 下的：

```yaml
temperature: 0.10
prototype_ema: 0.95
semantic_weight: 0.65
affordance_weight: 0.30
intrinsic_weight: 0.05
heatmap_topk_fraction: 0.05
stage_low_quantile: 0.30
stage_high_quantile: 0.70
boundary_margin: 0.05
```

Proto horizons 仍由：

```yaml
horizons: [1, 2, 4, 8, 15]
horizon_weights: [1.0, 1.0, 0.75, 0.5, 0.25]
```

控制。

## 8. 新增日志

主要看：

```text
proto_known_fraction
proto_unknown_fraction
proto_initialized
proto_acc_h1/h2/h4/h8/h15
proto_num_h*
proto_future_cosine_h*
proto_cf_margin_h*
proto_response_gap_h*
proto_loss
proto_weighted_loss
```

解释：

- `proto_acc_h15`：标准 Dreamer latent 经过真实动作前缀预测 15 步后，能否正确识别 future task-evidence 类别；
- `proto_cf_margin_h15`：正确动作前缀是否比打乱动作更接近真实 posterior future；
- `proto_response_gap_h15`：换动作后 full latent future 是否真正发生变化；
- `proto_future_cosine_h15`：Dreamer 原生 15-step prior 与 replay posterior future 的投影余弦，仅诊断，不进 loss。

## 9. 公平性

对 `ab_dreamer` vs `ab_dreamer_proto`，建议保持以下参数完全相同：

- seed；
- environment task；
- total steps；
- batch size / length；
- RSSM size；
- actor/critic；
- MineCLIP intrinsic reward；
- evaluation protocol。

唯一训练目标差异应是：

```text
+ proto_scale * GenericProtoLoss
```

