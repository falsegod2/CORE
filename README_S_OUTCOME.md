# ISO3 S-Aff + Fixed S5A + NoInverse + S-Outcome + Diagnostics

这是一个严格的 **S-Outcome 单变量消融版本**。基线来自
`ISO3-SAff-SOnly5A-NoInverse-Diagnostics`，不加入 RandomHorizon。

## 1. 保持不变

- `inverse_loss_scale: 0.0`
- `affordance_s_scale: 1.0`
- `z_action_adv_scale: 0.0`
- Fixed S5A：`h={1,2,4,8,15}`，weights=`[1,1,.75,.5,.25]`，scale=`0.05`
- Counterfactual Action Diagnostics
- posterior re-anchor / Direct-vs-Composed surrogate diagnostics
- Actor 的 imagined reward 定义完全不变

## 2. 新增 S-Outcome

对与 S5A 相同的起点和 horizon，复用原 S5A 已经 rollout 得到的
`S_hat_{t+k}`，不额外采样 RSSM：

`[S_t, S_hat_{t+k}] -> outcome_head -> symlog(G_t^k)`

其中 target 只来自 replay 的真实环境 reward：

`G_t^k = sum_{j=1..k} discount^(j-1) * reward_{t+j}`

跨 episode reset 的 prefix 由已有 `is_first` mask 排除。

默认损失：

`L_out = weighted_mean_h MSE(pred_symlog_return, symlog(real_k_step_return))`

总 world-model loss 额外增加：

`+ 0.01 * L_out`

S-Outcome **不进入 Actor reward**，因此不是 intrinsic reward / reward shaping。

## 3. 默认配置

```yaml
s_multi_step_consistency:
  enabled: true
  horizons: [1, 2, 4, 8, 15]
  horizon_weights: [1.0, 1.0, 0.75, 0.5, 0.25]
  projection_dim: 512
  starts_per_sequence: 4
  projection_seed: 314159
  loss_scale: 0.05

  outcome:
    enabled: true
    loss_scale: 0.01
    hidden_dim: 256
    init_seed: 161803
    discount: 0.997
    positive_weight: 1.0
```

`positive_weight=1.0` 是刻意保持的第一版：不人为重采样/放大成功轨迹，先测试
“真实多步 outcome grounding 本身”有没有独立收益。

## 4. 新日志

每个 `h=1,2,4,8,15`：

- `s_outcome_loss_h*`
- `s_outcome_target_return_mean_h*`
- `s_outcome_target_return_std_h*`
- `s_outcome_target_nonzero_frac_h*`
- `s_outcome_pred_return_mean_h*`
- `s_outcome_mae_h*`
- `s_outcome_corr_h*`

整体：

- `s_outcome_loss`
- `s_outcome_scale`
- `s_outcome_weighted_loss`
- `s_outcome_enabled`
- `s_outcome_discount`
- `s_outcome_positive_weight`

原有 `s_multistep_*`、`s_cf_*`、`s_dvc_*` 全部保留。

## 5. 重点看什么

首先看 `s_outcome_target_nonzero_frac_h15`。如果长期接近 0，说明真实 reward 太稀疏，
S-Outcome 没获得有效监督，此时不能只看 outcome loss 下降就认为学会了。

当 nonzero fraction 足够后，重点联合看：

- `s_outcome_corr_h15`：预测 outcome 与真实 k-step return 是否同向；
- `s_outcome_mae_h15`：绝对预测误差；
- `s_cf_margin_shuffle_h15`：action fidelity 是否仍健康；
- `s_dvc_reanchor_gain_h15`：open-loop drift；
- `imag_reward_* -> value_std -> actor_grad_norm -> eval_success`：outcome grounding 是否真正转化到控制链。

## 6. RNG / 严格消融

S-Outcome head 使用独立、可恢复的 forked CPU RNG 初始化，因此增加该 head 不会移动原模型的
全局初始化 RNG 状态。S-Outcome 直接复用 S5A 已有 stochastic rollout，不额外调用
`img_step_s(sample=True)`，因此不会额外消耗训练 rollout RNG。

可运行：

```bash
python test_s5a_diagnostics_invariance.py
python test_s_outcome.py
python audit_s_outcome.py
```
