from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = {
    "task patch encoder": "class TaskConditionedPatchEncoder" in (ROOT / "networks.py").read_text(),
    "task embedding observation": '"task_embedding"' in (ROOT / "envs/tasks/base/agent_wrapper.py").read_text(),
    "task patch config": "task_patch_fusion:" in (ROOT / "configs.yaml").read_text(),
    "no global visual embedding": "mineclip_embedding" not in (ROOT / "configs.yaml").read_text(),
    "no heatmap field": "cnn_keys: '^(image|heatmap)$'" not in (ROOT / "configs.yaml").read_text(),
    "no long-term config": "long_term" not in (ROOT / "configs.yaml").read_text().lower(),
    "single RSSM": "class RSSM" in (ROOT / "networks.py").read_text(),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("AUDIT FAILED: " + ", ".join(failed))
print("AUDIT PASSED: DreamerV3 + frozen task text + top-K RGB patch fusion.")
