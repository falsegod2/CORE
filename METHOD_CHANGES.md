# ISO3-RGB-Aux-NoLong：修改说明

本版本以 `ISO2-M-NoLong` 为主干。目标不是删除 affordance map 本身，而是阻止它作为观测直接进入世界模型 encoder，将其改为训练期辅助教师信号。

## 1. RGB-only observation encoder

原配置：

```yaml
encoder.cnn_keys: '^(image|heatmap)$'
```

现配置：

```yaml
encoder.cnn_keys: '^image$'
```

因此 posterior 的视觉 embedding 只由 RGB 图像产生：

```text
RGB image -> ConvEncoder -> shared embedding -> S/Z posterior
```

`heatmap` 仍保留在 replay buffer 中，但只用于：

- full world-model decoder 的辅助重建目标；
- S-only affordance reconstruction；
- 环境端已有的 intrinsic reward 计算。

它不再进入 actor/policy 的在线视觉编码路径。

## 2. 修复 inverse dynamics 的时间错位

Replay 中初始状态的 action 为零；索引 `t` 的 action 是到达索引 `t` observation 的动作。因此状态对：

```text
(state[t-1], state[t])
```

必须监督：

```text
action[t]
```

旧代码错误使用 `data['action'][:, :-1]`，现改为：

```python
true_action_onehot = data['action'][:, 1:]
```

并使用 `is_first[:, 1:]` 屏蔽跨 episode/reset 的无效状态对。

## 3. 将 inverse MSE 改为离散动作交叉熵

MineDojo 动作为 one-hot 离散动作。旧 MSE 存在“全零预测也能得到较小损失”的问题。现将 `_inverse_dynamics` 输出解释为 logits，使用交叉熵：

```python
loss_inverse = cross_entropy(action_logits, action_index)
```

新增日志：

- `action_acc_s`：S 分支动作识别准确率；
- `loss_inverse`：正确对齐后的动作分类损失。

## 4. 对抗抑制 Z 分支动作泄漏

虽然 Z prior 不输入 action，但 Z posterior 仍能通过当前图像读到动作造成的视觉变化。新版本加入 Gradient Reversal action adversary：

```text
(z[t-1], z[t]) -> action classifier
```

classifier 尝试识别 action；gradient reversal 反向推动 Z 隐藏动作信息。

配置：

```yaml
z_action_adv_scale: 0.05
z_action_grl_scale: 1.0
```

新增日志：

- `loss_z_action_adv`；
- `action_acc_z`。

对于 12 个动作，随机分类准确率约为 `1/12 = 8.33%`。理想趋势是：

```text
action_acc_s 显著升高
action_acc_z 接近随机水平
```

可通过 `--z_action_adv_scale 0` 做关闭该模块的消融。

## 5. 分支独立 free-bits

旧实现先求 `KL_s + KL_z` 再统一截断，可能出现一个分支承担大部分 KL、另一个分支塌缩。现对两个分支分别应用一半的 free-bits，再求和。

新增日志：

- `kl_s`；
- `kl_z`。

这样可以直接观察两个分支是否都在使用 latent capacity。

## 6. 评估 episode 数改为 10

旧实验每次只评估 3 个 episode，成功率只能取 0、1/3、2/3、1，方差过大。默认改为：

```yaml
eval_episode_num: 10
```

## 推荐实验矩阵

所有实验固定 3 seeds、1M environment steps、每次 10 eval episodes。

| 方法 | encoder | S-affordance | Z adversary | 目的 |
|---|---|---:|---:|---|
| Old ISO2-M-NoLong | RGB+heatmap | 2.0 | 无 | 原始对照 |
| RGB-only basic | RGB | 0 | 0 | 验证删除 heatmap 输入本身 |
| RGB-only + S-aff | RGB | 1.0 | 0 | 验证 affordance teacher 作用 |
| ISO3 Full | RGB | 1.0 | 0.05 | 验证完整解耦改进 |

推荐命令覆盖：

```bash
# RGB-only basic
python expr.py --configs minedojo --task minedojo_harvest_log_in_plains \
  --affordance_s_scale 0 --z_action_adv_scale 0 --logdir ./logdir

# RGB-only + S-affordance
python expr.py --configs minedojo --task minedojo_harvest_log_in_plains \
  --affordance_s_scale 1.0 --z_action_adv_scale 0 --logdir ./logdir

# Full
bash ./scripts/train.sh harvest_log_in_plains
```

## 训练时必须检查的日志

1. 启动输出必须出现：

```text
Encoder CNN shapes: {'image': (64, 64, 3)}
```

不得出现 `heatmap`。

2. `action_acc_s` 应高于 `action_acc_z`。

3. `kl_s` 与 `kl_z` 不应长期接近 0。

4. 比较成功率时优先报告：

- success AUC；
- 最后 20 次评估均值；
- 800K–1M 平均成功率；
- 达到稳定成功率阈值所需步数。

## Checkpoint兼容性

该版本将视觉 encoder 输入通道从 `RGB+heatmap = 4` 改为 `RGB = 3`，并新增 `z_action_adversary` 参数，因此旧版 `latest.pt` 与 optimizer state 不能直接完整恢复。正式实验应使用新 logdir 从头训练。若仅做权重迁移，需要手工处理第一层卷积和新增模块，不能用于公平主结果。
