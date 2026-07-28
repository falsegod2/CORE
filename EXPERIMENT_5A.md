# 实验五A：Multi-step RSSM Consistency

## 1. 实验目的

标准 DreamerV3 的逐步 KL 主要约束单步 prior 与 posterior。Actor 训练时却从一个 posterior 起点连续展开多步 prior，中间没有真实观测校正。实验五A直接约束这种 open-loop rollout：

\[
\hat s_{t+k}^{p}=F_\phi^{(k)}(s_t^q,a_{t+1:t+k})
\]

应与 replay 中真实观测得到的未来 posterior：

\[
s_{t+k}^{q}
\]

在同一固定投影空间中保持一致。

## 2. 实验边界

五A只增加一个辅助损失，不增加：

- MineCLIP 表示输入；
- 新 latent 分支；
- object/slot/prototype；
- intrinsic reward；
- long-term jump；
- Actor/Critic 新输入；
- 并行动作前缀预测器（这是五B）。

因此它检验的变量只有一个：**显式多步 open-loop latent 监督是否改善 RSSM imagination。**

## 3. 目标表示

对离散 RSSM，不匹配随机采样的 one-hot，而使用类别概率和确定性状态：

\[
\bar f_t=[\operatorname{softmax}(\ell_t),h_t].
\]

再经过不训练的固定随机投影：

\[
y_t=\operatorname{norm}(R\bar f_t).
\]

固定投影使用独立局部随机数生成器，不消耗全局 RNG，因此不会改变原始 Dreamer 参数初始化顺序。

## 4. 损失

默认监督距离：

\[
\mathcal K=\{1,2,4,8,15\}.
\]

\[
\mathcal L_{5A}
=
\frac{1}{\sum_k w_k}
\sum_{k\in\mathcal K}w_k
\left(1-\cos(\hat y_{t+k}^{p},\operatorname{sg}(y_{t+k}^{q}))\right).
\]

总损失：

\[
\mathcal L
=
\mathcal L_{DreamerV3}+0.05\mathcal L_{5A}.
\]

默认权重为：

```yaml
horizons: [1, 2, 4, 8, 15]
horizon_weights: [1.0, 1.0, 0.75, 0.5, 0.25]
```

## 5. 计算控制

为了避免把训练成本放大到不可接受的程度，每条长度32的 replay sequence 只选4个均匀分布的起点，然后一次展开15步，并在指定 horizon 读取损失。跨 episode boundary 的样本通过 `is_first` 自动屏蔽。

## 6. 启动

```bash
cd CORE-DREAMERV3-MULTISTEP-RSSM-5A
conda activate ls_imagine
python scripts/audit_multistep5a.py
bash ./scripts/train_5a.sh harvest_log_in_plains 0 0
```

参数依次为：任务名、seed、物理GPU编号。

## 7. 新日志指标

- `multistep_loss`：加权总一致性损失；
- `multistep_loss_h1/h2/h4/h8/h15`：各预测距离误差；
- `multistep_cosine_h*`：各距离 prior/posterior 相似度；
- `multistep_valid_h*`：没有跨 episode boundary 的有效样本比例；
- `multistep_target_std`：目标投影标准差，异常接近0时需排查表示退化；
- `base_model_loss`：不含五A的原Dreamer世界模型损失；
- `model_loss`：加入五A后的总损失。

## 8. 第一阶段停止规则

先跑到300k，不直接承诺百万步。继续训练需要同时满足：

1. `h8`、`h15` 的 cosine 随训练上升，而不仅是 `h1`；
2. `base_model_loss`、image loss、KL 没有持续恶化；
3. 同步数 evaluation 不低于纯 DreamerV3；
4. FPS 下降可接受；
5. 没有出现长距离损失主导总梯度的迹象。

若 `h1` 改善但 `h8/h15` 不改善，说明仍然只学到局部一步一致性。若 latent 指标改善但控制性能无提升，则五A只能作为机制消融，不能作为主创新。
