# DreamerV3 + 模块化消融基线

这个工程把 **纯 DreamerV3** 与当前方法的各个模块放进同一份代码，通过 `configs.yaml` 的开关/预设做消融，不需要维护多份工程。

## 1. 核心原则

### 纯 Dreamer

`ab_dreamer` 使用上传的 `DreamerV3_Clean_PixelOnly_BinaryReward` 中的 **原始单流 RSSM**：

- 单一 `stoch/deter`；
- RGB-only encoder/decoder；
- 不含 Dual S/Z；
- 不含 S-Aff；
- 不含 S5A；
- 不含 Balanced Outcome；
- 不含 Prototype Utility。

为了与当前已有实验协议一致，默认仍保留原 Dreamer 基线中的 MineCLIP/intrinsic shaping：

```yaml
intrinsic_reward_scale: 1.0
```

如果要做环境奖励-only：

```bash
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer_no_intrinsic 0
```

### S 分支模块的依赖与 Proto 例外

S-Aff、S5A、Outcome 仍然定义在 controllable `S` branch 上，因此必须先有 Dual S/Z。

Generic Proto 现在增加了一个**严格的单流 Dreamer 消融**：

- `ab_dreamer_proto` = Dreamer 单一 RSSM + Generic Proto(full latent)
- `ab_proto` / `ab_dual_proto` = Dreamer + Dual S/Z + Generic Proto(S branch)

因此可以做完整 2×2：Dreamer / Dreamer+Proto / Dreamer+Dual / Dreamer+Dual+Proto。

`ab_dreamer_proto` 不会强行创建 S branch；它使用标准 Dreamer `dynamics.img_step` 对 full latent 做 action-prefix rollout，并且不加入 S5A consistency loss。

## 2. 推荐的核心消融矩阵

| Profile | Single RSSM | Dual S/Z | S-Aff | S5A loss | Outcome | Generic Proto |
|---|---:|---:|---:|---:|---:|---:|
| `ab_dreamer` | ✓ |  |  |  |  |  |
| `ab_dreamer_proto` | ✓ |  |  |  |  | ✓ (full latent) |
| `ab_dual` |  | ✓ |  |  |  |  |
| `ab_saff` |  | ✓ | ✓ |  |  |  |
| `ab_s5a` |  | ✓ |  | ✓ |  |  |
| `ab_outcome` |  | ✓ |  | **0** | ✓ |  |
| `ab_proto` |  | ✓ |  | **0** |  | ✓ |
| `ab_saff_s5a` |  | ✓ | ✓ | ✓ |  |  |
| `ab_saff_s5a_outcome` |  | ✓ | ✓ | ✓ | ✓ |  |
| `ab_saff_s5a_proto` |  | ✓ | ✓ | ✓ |  | ✓ |
| `ab_full` |  | ✓ | ✓ | ✓ | ✓ | ✓ |

`ab_outcome` / `ab_proto` 中，S5A **rollout machinery** 会被调用，因为 Outcome/Proto 需要 `[S_t, S_hat_{t+k}]`；但是：

```yaml
s_multi_step_consistency.loss_scale: 0.0
```

所以它们不会偷偷加入 S5A consistency loss。这是为了真正测单模块效果。

## 3. 运行

### Tree

```bash
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dreamer_proto 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dual 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_dual_proto 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_saff 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_s5a 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_outcome 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_proto 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_saff_s5a 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_saff_s5a_proto 0
bash ./scripts/train_ablation.sh harvest_log_in_plains ab_full 0
```

### Sheep

把任务名换成：

```bash
shear_sheep
```

例如：

```bash
bash ./scripts/train_ablation.sh shear_sheep ab_proto 0
```

第三个参数是 seed。

## 4. 直接改参数

最常用参数全部放在 `configs.yaml -> defaults` 顶部：

```yaml
module_dual_sz: false
module_saff: false
module_s5a: false
module_outcome: false
module_proto: false
module_inverse: false
module_z_adv: false

saff_scale: 1.0
full_heatmap_scale: 1.0
s5a_scale: 0.05
outcome_scale: 0.01
proto_scale: 0.01
inverse_scale: 1.0
z_adv_scale: 1.0
intrinsic_reward_scale: 1.0
proto_label_mode: task_evidence
proto_latent_source: auto   # full for single Dreamer, s for Dual S/Z
```

也可覆盖 flat 参数，例如：

```bash
python expr.py \
  --configs minedojo ab_full \
  --task minedojo_harvest_log_in_plains \
  --proto_scale 0.005 \
  --seed 0 \
  --logdir ./logdir
```

### Prototype 权重消融

已有：

```bash
... --configs minedojo ab_full_proto_half
```

对应：

```yaml
proto_scale: 0.005
```

### 去 Outcome

直接使用：

```bash
... --configs minedojo ab_saff_s5a_proto
```

这正是：

`Dual + S-Aff + S5A + Generic Proto, w/o Balanced Outcome`。

## 5. 隐藏辅助损失防护

这个版本专门修了一个容易导致“假消融”的问题：

当 `module_saff=false` 时，runtime 会强制：

```yaml
decoder.cnn_keys: '^image$'
full_heatmap_scale: 0
```

因此不会出现：

> 表面关闭 S-Aff，但 full latent decoder 仍在偷偷预测 heatmap。

只有 S-Aff 开启时，decoder 才变成：

```yaml
decoder.cnn_keys: '^(image|heatmap)$'
```

## 6. Heatmap teacher 的计算也按需打开

- Dreamer / Dual / S5A / Outcome：默认不运行 affordance U-Net；
- S-Aff：运行 heatmap pipeline；
- Generic Proto：如果 `affordance_weight > 0`，运行 heatmap pipeline。

这样纯 Dreamer 不会因为“虽然不用 heatmap、但环境里仍在算 U-Net”而平白增加大量 wall-clock 开销。

LSImagineWrapper 仍保持统一 observation schema，heatmap 不需要时输出零，因此 replay 结构在不同配置间保持兼容。

## 7. 两套 RSSM backend 是真正分开的

为了避免“把所有 loss 设成 0 但底层仍然是 Dual RSSM”这种错误基线，本工程保留两套真实 backend：

- `models_dreamer.py + networks_dreamer.py`：上传的 clean single-stream DreamerV3；
- `models_modular.py + networks_modular.py`：当前 Dual S/Z 方法。

`models.py` 只是一个 lazy dispatcher。

因此：

```text
ab_dreamer
```

真的使用单流 Dreamer RSSM；而不是 Dual S/Z + 所有 auxiliary loss=0。

## 8. 建议论文优先跑哪些

如果算力有限，最小但信息量高的一组：

1. `ab_dreamer`
2. `ab_dual`
3. `ab_saff`
4. `ab_s5a`
5. `ab_proto`
6. `ab_saff_s5a`
7. `ab_saff_s5a_proto`
8. `ab_full`

这组可以回答：

- Dual S/Z 本身有多少贡献？
- S-Aff 单独贡献多少？
- S5A 单独贡献多少？
- Generic Proto 单独贡献多少？
- S-Aff + S5A 是否互补？
- Proto 在完整 representation chain 上还能增加多少？
- Outcome 是否真的必要？

## 9. 实验公平性

不同 profile 不要互相 resume checkpoint。

保持一致：

```text
seed
steps
prefill
batch_size
batch_length
train_ratio
imag_horizon
eval_every
eval_episode_num
reward protocol
```

正式论文建议至少 seed 0/1/2。

## 10. 自检

```bash
PYTHONPATH=. python scripts/audit_ablation_profiles.py
python -m compileall -q .
```

`audit_ablation_profiles.py` 会检查核心 profile 是否真的打开/关闭预期模块，以及 S-Aff 关闭时是否仍残留 heatmap decoder loss。
