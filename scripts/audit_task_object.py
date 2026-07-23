from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
networks = (ROOT / "networks.py").read_text()
models = (ROOT / "models.py").read_text()
configs = (ROOT / "configs.yaml").read_text()
checks = {
    "task object encoder": "class TaskRelevantObjectEncoder" in networks,
    "task-conditioned object queries": "_task_to_objects" in networks,
    "task relevance candidates": "candidate_topk" in configs,
    "object coverage auxiliary": "task_object_coverage" in networks and "get_aux_losses" in models,
    "task embedding observation": '"task_embedding"' in (ROOT / "envs/tasks/base/agent_wrapper.py").read_text(),
    "global semantic teacher observation": '"mineclip_embedding"' in (ROOT / "envs/tasks/base/agent_wrapper.py").read_text(),
    "semantic alignment auxiliary": "task_object_semantic_align" in networks,
    "no pure slot attention": "SlotAttention" not in networks,
    "no heatmap input": "cnn_keys: '^(image|heatmap)$'" not in configs,
    "no long-term config": "long_term" not in configs.lower(),
    "single RSSM": "class RSSM" in networks,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("AUDIT FAILED: " + ", ".join(failed))
print(
    "AUDIT PASSED: DreamerV3 + frozen MineCLIP text/video semantic teachers "
    "+ task-guided object tokens; no pure slots, heatmap, or long-term path."
)
