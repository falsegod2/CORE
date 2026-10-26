> 2026-10-21 新入口与消融说明：请先阅读 [README_ABLATIONS_CN.md](README_ABLATIONS_CN.md)。本文下方保留历史版本说明，历史测试数量与启动目录不代表当前包。

# Evidence 移植说明

基底：`LS-Imagine_Clean_PixelAffordance_BinaryReward(2).zip`。

基底 ZIP SHA256：`ec53107a5bcf7d1312ee28ff07f802253674061eba04ce530bba82523fc56002`。

Evidence 来源：`WM_Experiment_Suite_EvidenceDreamer_20261014(1).zip` 中的 Evidence 方法、dual RSSM 和固定版本 PyTorch DreamerV3 后端。来源 ZIP SHA256：`bebc104e63ac28be11c9f5b27b8dcad8a91e2506ed4b6a95ee8f9ccbc4373a49`。

此处只复用方法核心与训练状态管理。旧 WM 的环境、RPC、WorldPlan、固定地图、原生奖励额外加分、评估种子和模拟器生命周期均没有移入。

## 文件组织

- `envs/tasks/minedojo/__init__.py`：唯一修改的原包文件；`evidence_specs=None` 的默认 LS 路径保留，Evidence 显式启用训练用 heatmap-only teacher，并用绝对路径读取现有权重。
- `clean_evidence/environment.py`：对原 Clean 环境工厂和外层 action wrappers 做数据包桥接；保留原任务边界、discount、success 计数。
- `clean_evidence/semantic_teacher.py`：只产生语义监督标签，不触发 LS concentration shaping 或 zoom。
- `clean_evidence/methods/`：Evidence replay、S/Z 动力学、观察记忆、FC/RC、actor–critic 及 learner checkpoint。
- `clean_evidence/vendor/dreamerv3_torch/`：固定后端与其 LICENSE；包内导入隔离，避免误导入根目录的 LS 同名模块。
- `clean_evidence/configuration.py` 与 `configs/evidence.yaml`：Clean 任务/运行参数加 Evidence 正式模型配置，避免把 LS 专属 heads 当成 Evidence heads。
- `clean_evidence/runner.py`：串行训练、持续环境复用、同任务评估、累计预算和保存恢复。
- `clean_evidence/storage.py`：完整状态的原子保存、校验、保留和防覆盖。
- `clean_evidence/runtime/`：检查已有 Python/Java8/CUDA/权重与本地 tokenizer。
- `run_evidence.py` / `run_evidence.sh`：训练、恢复、检查、独立评估入口。
- `evidence_tests/`：协议、模型、训练管理和恢复验证。

## 保持的模型机制

正式配置仍为 deter=4096、hidden=1024、stoch=32×32，RGB CNN depth=96，memory 16 observations×4 tokens，memory_dim=256，burnin=32。局部/后验 heatmap 监督、FC 多时间跨度与 RC 有限真实任务奖励约束保留；想象过程中固定观察记忆，不把想象帧写入真实观察记忆。

适配变化仅包括包内命名空间、环境字段桥接、Clean transition discount 优先级、Clean MineCLIP 信号口径和串行训练调度。模型不接收 ground-truth heatmap、成功标签或未来观察作为策略输入。

Clean 的成功/超时可以同时为 `is_last=True`、`is_terminal=False`，因此 actor 想象起点同时排除两类结束边界；保留环境原始标记，不把已结束 episode 的末行当成可继续交互的起点。FC/RC 的真实轨迹边界检查保持有效。

## 与旧训练管理的差异

Evidence 新入口精确停止在累计 `--steps`，按累计阈值定期保存。恢复读完整快照并拒绝同名覆盖，不会从固定 100,000 步重新计数。训练时评估沿用 Clean 的初始化评估和串行节奏；物理模拟器在单次运行内不反复关闭重建。恢复一个已退出的进程时，模拟器仍需要正常重新启动，且开始新 episode。

原始 LS `expr.py`、`test.py`、`tools.py`、`models.py`、`configs.yaml` 没有修改，因此 LS 原有训练管理行为仍由其原入口负责。
