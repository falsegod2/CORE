from pathlib import Path

root = Path(__file__).resolve().parent
checks = []

def has(path, text):
    return text in (root / path).read_text()

checks.append(("S-Aff scale = 1", has("configs.yaml", "affordance_s_scale: 1.0")))
checks.append(("Inverse scale = 1", has("configs.yaml", "inverse_loss_scale: 1.0")))
checks.append(("Z adversary disabled", has("configs.yaml", "z_action_adv_scale: 0.0")))
checks.append(("S-only 5A enabled", has("configs.yaml", "s_multi_step_consistency:")))
checks.append(("5A scale = 0.05", has("configs.yaml", "loss_scale: 0.05")))
checks.append(("S-only transition exists", has("networks.py", "def img_step_s")))
checks.append(("S-only loss imported", has("models.py", "import s_multistep_consistency")))
checks.append(("S-only loss called", has("models.py", "self._s_multistep(")))
checks.append(("No GRU init fix in strict build", not has("networks.py", "self._cell_s.apply(tools.weight_init)")))

failed = False
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok
raise SystemExit(1 if failed else 0)
