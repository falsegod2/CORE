from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
networks = (ROOT / "networks.py").read_text()
models = (ROOT / "models.py").read_text()
configs = (ROOT / "configs.yaml").read_text()
checks = {
    "task object encoder": "class TaskRelevantObjectEncoder" in networks,
    "task-conditioned object queries": "_task_to_objects" in networks,
    "balanced Sinkhorn assignment": "_sinkhorn_assignment" in networks,
    "object competition": "competition = assignment /" in networks,
    "softened task prior": "assignment_uniform_mix" in configs,
    "reduced relevance bias": "relevance_bias: 0.25" in configs,
    "two-object diagnostic": "num_objects: 2" in configs,
    "global-only semantic teacher": "_global_semantic(global_token)" in networks,
    "no pooled object semantic target": "objects.mean(dim=-2)" not in networks,
    "competition entropy auxiliary": "task_object_competition_entropy" in networks,
    "feature diversity auxiliary": "task_object_feature_diversity" in networks,
    "task embedding observation": '"task_embedding"' in (ROOT / "envs/tasks/base/agent_wrapper.py").read_text(),
    "global semantic observation": '"mineclip_embedding"' in (ROOT / "envs/tasks/base/agent_wrapper.py").read_text(),
    "no pure slot attention": "SlotAttention" not in networks,
    "no heatmap input": "cnn_keys: '^(image|heatmap)$'" not in configs,
    "no long-term config": "long_term" not in configs.lower(),
    "single RSSM": "class RSSM" in networks,
    "generic auxiliary integration": "get_aux_losses" in models,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("AUDIT FAILED: " + ", ".join(failed))
print(
    "AUDIT PASSED: DreamerV3 + competitive task-relevant object tokens V2; "
    "balanced binding, global-only MineCLIP alignment, no heatmap or long-term path."
)
