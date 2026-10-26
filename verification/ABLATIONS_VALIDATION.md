# 验证记录：2026-10-21

本包基于 `LS_Imagine_Clean_Evidence_20261020.zip`。默认正式 checkpoint 间隔为 20,000，原始任务 YAML 与世界重置 wrapper 保留。

## 自动验证结果

- 95 项测试通过，0 失败；13.72 秒，83 条依赖/弃用警告。完整逐项记录见 `ablation_test_results.xml`。
- 七个 Evidence 配置均执行实际小网络的 world-model、actor、critic 更新及状态恢复；检查关闭组件确实不参与对应计算。
- 同 seed 下七组共有 world-model、actor、value、slow value 初始化参数逐项一致。
- `no_semantic` 不构造 U-Net teacher，不要求其权重，也不读取热图训练目标。
- LS 执行真实小模型的无 zoom、有 zoom、空和非空长时想象分支；验证 12 个物理动作与内部第 13 个虚拟动作的区别。
- LS 串行 runner 完成 16 步训练、保存、恢复至 24 步，仅新增 8 次训练动作；在 10/20 步按计划保存。
- 原 Clean wrapper 配合假物理/teacher，验证 HUD、奖励、动作、终止以及六次 reset 中五次 FastReset；训练/评估 LS 均保留 Concentration。
- 检查错误覆盖、残留 checkpoint、丢失元信息、GPU 命令解析、八类实验输出隔离及串行计划。
- 检查 RC 稀少正样本的加权统计及动作计数恢复；新增想象奖励诊断前后 actor/critic 参数与 RNG 一致。
- 69 个 Python 文件通过 Python 3.9 语法检查；3 个 Bash 入口通过 `bash -n`。

本地验证环境：Python 3.12.14、PyTorch 2.5.1 CPU、NumPy 1.26.4、Gym 0.21 原源码、ruamel.yaml 0.17.40。Gym 的插件元数据接口在测试中适配 Python 3.12；项目运行目标仍是原有 Python 3.9 / `ls_imagine` 环境。私有测试依赖未打入交付包。

## LS 修复的边界

`models.py` 跳过没有被选中状态的长时想象分支，避免对空批次构造 Categorical；非空分支保持原计算。`expr.py` 跳过没有新指标的空日志窗口；`tools.py` 把单值标准差记为 0、按需导入 wandb，并允许保存原 replay 采样器的 RNG。详细差异见 `clean_ls_changes.patch`。

新 `clean_ls` runner 只管理固定目录、预算、串行评估、已有资源路径和完整 checkpoint。原 LS 的 encoder/decoder 输入、奖励、zoom 标签、长时想象目标与环境 wrapper 未被替换为 Evidence。为了可靠恢复，两种新入口均 `compile=false`。

源码/配置/权重身份不同会拒绝 resume。Minecraft 世界不在 checkpoint 中；恢复会开始新 episode，尚未完成的 LS 尾段不继续，未完成的 zoom 标签保持无效。旧代码 checkpoint 应留在旧实验目录中，不宣称跨版本直接兼容。

## 尚未验证

没有访问用户远程主机，未执行其 Minecraft/Java/Gradle 启动、GPU 正式尺寸更新或百万步实验。测试中的底层物理和 teacher 是替身；模型更新使用真实 PyTorch 实现。95 项通过表示上述代码路径已验证，不能保证服务器外部依赖不会出错，也不代表消融或完整方法有效性已经建立。

`20261020_baseline/` 保存前一版本的历史验证，不能用其中的 57 项数字代替本版本记录。
