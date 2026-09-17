# Stable V1.1

Stable V1.1 直接建立在 Stable V1 上，不改 Dual S/Z、Full Heatmap、S-only Affordance、S5A、Outcome evidence gate，也不改 Proto 的总 nominal scale=0.01。

唯一核心变化：给 Generic Proto 增加 **teacher reliability gate**，避免 Sheep 等弱 task-evidence 阶段仍以满强度使用 Low/Mid/High pseudo-label 塑形 S。

## 1. Reliability score

每次 world-model update 使用当前 batch 的 teacher 区分度：

`R_batch = 0.65 * clip(semantic_span,0,1) + 0.30 * clip(affordance_span,0,1)`

其中 span 是当前 batch 的 q05-q95 原始跨度，不使用归一化后的相对 Low/Mid/High 分布。

为了避免单个幸运 batch 突然打开 Proto，R 使用从 0 开始的 EMA：

`R_ema = 0.99 * R_ema + 0.01 * R_batch`

默认 gate：

`g_proto = clip((R_ema - 0.30) / (0.60 - 0.30), 0, 1)`

实际 Proto loss：

`L_proto_actual = g_proto * L_proto`

WorldModel 外层仍保留 `s_prototype_scale=0.01`，因此实际有效权重为：

`0.01 * g_proto`。

## 2. Prototype bank protection

当 `g_proto == 0` 时，Low/Mid/High pseudo-label 不更新 prototype bank，避免弱 teacher 在早期污染 prototype geometry。

真实 Success 是环境硬标签，因此仍然累计 Success support/prototype；但 Success class 只有同时满足：

- `success_support >= 256`
- `g_proto >= 0.10`

才参与 Proto logits / CE。

## 3. 保持 Stable V1 不变的部分

- Full Heatmap：保持历史最强版原设置。
- S-Aff：1.0。
- S5A：0.05，horizons `[1,2,4,8,15]`。
- Proto nominal scale：0.01。
- Outcome nominal scale：0.01。
- Outcome evidence gate：保持 Stable V1。
- Proto EMA quantile、prototype_ema=0.98、boundary_margin=0.10、success_min_support=256：保持 Stable V1。
- Inverse / Z adversary：继续关闭。

## 4. 新增直观日志

除 Stable V1 的 health 指标外，新增：

- `health/proto_reliability_batch`：当前 batch teacher 区分度。
- `health/proto_reliability_ema`：稳定化后的 teacher 区分度。
- `health/proto_gate`：当前 Proto 真正开启比例 `[0,1]`。
- `health/proto_effective_scale`：实际 Proto world-model 权重 `0.01 * gate`。
- `health/proto_success_rel_ready`：Success 是否通过 reliability 条件。
- `health/proto_bank_update`：当前 Low/Mid/High pseudo prototype 是否允许更新。

控制台示例：

```
[HEALTH] S5A(cos/margin/gap)=.../.../... |
Proto(acc/known/gate/succReady)=.../.../.../... |
Rel(batch/ema/effScale)=.../.../... |
Outcome(gate/posEMA/corr)=.../.../... |
Policy(Rstd/Vstd/Agrad)=.../.../... | ALERT=0
```

## 5. 检查与启动

```bash
python audit_stable_v1_1.py
python test_stable_v1.py
python test_stable_v1_1.py
```

砍树：

```bash
bash ./scripts/train_stable_v1_1.sh harvest_log_in_plains 0
```

剪羊：

```bash
bash ./scripts/train_stable_v1_1.sh shear_sheep 0
```

seed1：

```bash
bash ./scripts/train_stable_v1_1.sh shear_sheep 1
```

输出目录为 `./logdir/stable_v1_1/`，不会覆盖 Stable V1。
