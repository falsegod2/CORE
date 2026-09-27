from pathlib import Path

root = Path(__file__).resolve().parent


def has(path, text):
    return text in (root / path).read_text()


checks = [
    ("Prototype module exists", (root / "s_prototype_utility.py").exists()),
    ("Generic task-evidence mode is default", has("configs.yaml", "label_mode: task_evidence")),
    ("No sheep-specific stage names in default config", not has("configs.yaml", "Search / Acquired / Ready")),
    ("MineCLIP raw task score is exposed", has("envs/tasks/base/clip_wrapper.py", "task_score")),
    ("Task score is carried through replay", has("envs/tasks/base/ls_imagine_wrapper.py", "'task_score'")),
    ("RGB encoder remains image-only", has("configs.yaml", "cnn_keys: '^image$'")),
    ("Task score is not added to encoder keys", "task_score" not in (root / "configs.yaml").read_text().split("encoder:", 1)[-1].split("decoder:", 1)[0]),
    ("Task-evidence labeler combines generic signals", has("s_prototype_utility.py", "build_task_evidence_labels") and has("s_prototype_utility.py", "heatmap_topk_evidence")),
    ("Flat signals can remain Unknown", has("s_prototype_utility.py", "labels = torch.full_like(rewards_bt, -1")),
    ("Real reward overrides pseudo stage", has("s_prototype_utility.py", "labels[success] = 3")),
    ("Posterior support detached/no_grad", has("s_multistep_consistency.py", "_prepare_prototype_support") and has("s_multistep_consistency.py", "@torch.no_grad()")),
    ("Uses existing S5A projector", has("s_multistep_consistency.py", "support_raw = self.projector") and has("s_multistep_consistency.py", "self.prototype_bank.loss")),
    ("Predicted future S is query", has("s_multistep_consistency.py", "self.prototype_bank.loss(\n                    pred_proj")),
    ("Prototype not actor reward", "s_prototype" not in (root / "models.py").read_text().split("class ImagBehavior", 1)[-1]),
    ("Prototype enters world-model loss", has("models.py", "+ self._s_prototype_scale * s_prototype_loss")),
]

failed = False
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok

raise SystemExit(1 if failed else 0)
