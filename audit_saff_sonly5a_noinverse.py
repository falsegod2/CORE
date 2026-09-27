from pathlib import Path
import re

root = Path(__file__).resolve().parent
cfg = (root / 'configs.yaml').read_text()
models = (root / 'models.py').read_text()
networks = (root / 'networks.py').read_text()

checks = {
    'inverse supervision disabled': bool(re.search(r'inverse_loss_scale:\s*0(?:\.0+)?\b', cfg)),
    'S-aff enabled': bool(re.search(r'affordance_s_scale:\s*1(?:\.0+)?\b', cfg)),
    'Z adversary disabled': bool(re.search(r'z_action_adv_scale:\s*0(?:\.0+)?\b', cfg)),
    'S-only multistep enabled': 's_multi_step_consistency:' in cfg and 'enabled: true' in cfg,
    '5A scale = 0.05': bool(re.search(r'loss_scale:\s*0\.05\b', cfg)),
    'S-only img step exists': 'def img_step_s(' in networks,
    'S multistep module is wired': '_s_multistep' in models and 's_multistep_loss' in models,
    'inverse head retained for strict RNG comparability': '_inverse_dynamics' in networks,
}

failed = False
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok

if failed:
    raise SystemExit(1)
print('ISO3-SAff + S-only 5A NoInverse static audit passed.')
