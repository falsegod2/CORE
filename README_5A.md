# CORE DreamerV3 — Experiment 5A

本目录在统一 RSSM、无 long-term jump、无 affordance intrinsic reward 的基线上加入 **Multi-step RSSM Consistency**。

核心修改只有：

- 新增 `multistep_consistency.py`；
- 在 `WorldModel._train()` 中增加 open-loop 多步 latent consistency；
- 新增 `multistep5a` 配置和训练脚本；
- 新增静态审计与CPU单元测试。

详细公式、指标和停止规则见 `EXPERIMENT_5A.md`。
