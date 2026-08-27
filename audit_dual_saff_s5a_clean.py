from pathlib import Path
import re, yaml
root = Path(__file__).resolve().parent
cfg_text = (root / 'configs.yaml').read_text()
expr = (root / 'expr.py').read_text()
models = (root / 'models.py').read_text()
mod = (root / 's_multistep_consistency.py').read_text()
net = (root / 'networks.py').read_text()
cfg = yaml.safe_load(cfg_text)['defaults']
tasks = yaml.safe_load((root/'envs/tasks/task_specs.yaml').read_text())
sms = cfg['s_multi_step_consistency']
checks = {
    'Dual S/Z transition exists': 'def img_step_s' in net and 'self._cell_z' in net and 'deter_z' in net,
    'S-Aff enabled': float(cfg['affordance_s_scale']) == 1.0,
    'Inverse disabled': float(cfg['inverse_loss_scale']) == 0.0,
    'Z adversary disabled': float(cfg['z_action_adv_scale']) == 0.0,
    'S-only S5A enabled': bool(sms['enabled']),
    'S5A horizons fixed': list(sms['horizons']) == [1,2,4,8,15],
    'S5A scale 0.05': abs(float(sms['loss_scale'])-0.05) < 1e-12,
    'No Outcome implementation in models': 's_outcome' not in models.lower(),
    'No Outcome implementation in S5A module': 'outcome' not in mod.lower(),
    'Encoder RGB only': cfg['encoder']['cnn_keys'] == '^image$' and cfg['encoder']['mlp_keys'] == '$^',
    'Decoder RGB+heatmap': 'image' in cfg['decoder']['cnn_keys'] and 'heatmap' in cfg['decoder']['cnn_keys'] and cfg['decoder']['mlp_keys'] == '$^',
    'RewardObs removed': 'env = wrappers.RewardObs(env)' not in expr,
    'Duplicate harvest reward removed': 'reward_specs' not in tasks['harvest_log_in_plains'],
    'Eval cadence 20k': int(cfg['eval_every']) == 20000,
    'Eval episodes 10': int(cfg['eval_episode_num']) == 10,
    'S5A diagnostics enabled': bool(sms.get('diagnostics',{}).get('enabled',False)),
    'Counterfactual diagnostics present': 's_cf_margin_shuffle_h' in mod,
}
failed=False
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed |= not ok
raise SystemExit(1 if failed else 0)
