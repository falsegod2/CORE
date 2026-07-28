#!/usr/bin/env python3
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
required = [
    root / "multistep_consistency.py",
    root / "models.py",
    root / "configs.yaml",
    root / "scripts" / "train_5a.sh",
]
missing = [str(x) for x in required if not x.exists()]
if missing:
    raise SystemExit("Missing files:\n" + "\n".join(missing))

models = (root / "models.py").read_text()
config = (root / "configs.yaml").read_text()
checks = {
    "module import": "import multistep_consistency" in models,
    "loss call": "self._multistep(" in models,
    "loss scale": "self._multistep_scale * multistep_loss" in models,
    "profile": re.search(r"^multistep5a:", config, re.M) is not None,
    "horizons": "horizons: [1, 2, 4, 8, 15]" in config,
}
for name, ok in checks.items():
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
if not all(checks.values()):
    sys.exit(1)
print("Experiment 5A static audit passed.")
