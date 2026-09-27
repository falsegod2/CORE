from pathlib import Path
import yaml

cfgs = yaml.safe_load(Path('configs.yaml').read_text())


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            recursive_update(base[key], value)
        else:
            base[key] = value

cfg = {}
recursive_update(cfg, cfgs['defaults'])
recursive_update(cfg, cfgs['minedojo'])
recursive_update(cfg, cfgs['stable_v1'])

sms = cfg['s_multi_step_consistency']
out = sms['outcome']
pro = sms['prototype_utility']

print('=' * 72)
print('Stable V1 configuration audit')
print('=' * 72)
print(f"S-Aff scale               : {cfg['affordance_s_scale']}")
print(f"S5A scale                 : {sms['loss_scale']}")
print(f"S5A horizons              : {sms['horizons']}")
print(f"Outcome scale             : {out['loss_scale']}")
print(f"Outcome evidence gate     : {out['evidence_gate_enabled']}")
print(f"Outcome gate EMA          : {out['evidence_gate_ema']}")
print(f"Outcome gate range        : [{out['evidence_gate_low']}, {out['evidence_gate_high']}]")
print(f"Proto scale               : {pro['loss_scale']}")
print(f"Proto EMA                 : {pro['prototype_ema']}")
print(f"Proto quantile mode       : {pro['quantile_mode']}")
print(f"Proto quantile EMA        : {pro['quantile_ema']}")
print(f"Proto boundary margin     : {pro['boundary_margin']}")
print(f"Success min support       : {pro['success_min_support']}")
print(f"Compact health log        : {cfg['compact_health_log']}")
print('=' * 72)
print('NOTE: Full heatmap reconstruction is unchanged from the strongest baseline.')
