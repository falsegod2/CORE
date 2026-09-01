from pathlib import Path

root = Path(__file__).resolve().parent

def has(path, text):
    return text in (root / path).read_text()

checks = [
    ("Prototype module exists", (root / "s_prototype_utility.py").exists()),
    ("Prototype enabled", has("configs.yaml", "prototype_utility:") and has("configs.yaml", "enabled: true")),
    ("Prototype scale 0.01", has("configs.yaml", "loss_scale: 0.01")),
    ("Prototype not actor reward", "s_prototype" not in (root / "models.py").read_text().split("class ImagBehavior", 1)[-1]),
    ("Posterior support detached/no_grad", has("s_multistep_consistency.py", "_prepare_prototype_support") and has("s_multistep_consistency.py", "@torch.no_grad()")),
    ("Uses existing S5A projector", has("s_multistep_consistency.py", "support_raw = self.projector") and has("s_multistep_consistency.py", "self.prototype_bank.loss")),
    ("Predicted future S is query", has("s_multistep_consistency.py", "self.prototype_bank.loss(\n                    pred_proj")),
    ("Zero reward can remain Unknown", has("s_prototype_utility.py", "labels = torch.full_like(distance, -1)")),
    ("Prototype enters world-model loss", has("models.py", "+ self._s_prototype_scale * s_prototype_loss")),
    ("Prototype metrics logged", has("models.py", 'metrics["s_prototype_weighted_loss"]')),
]

failed = False
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok

raise SystemExit(1 if failed else 0)
