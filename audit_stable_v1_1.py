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
for profile in ('defaults', 'minedojo', 'stable_v1', 'stable_v1_1'):
    recursive_update(cfg, cfgs[profile])

sms = cfg['s_multi_step_consistency']
out = sms['outcome']
pro = sms['prototype_utility']

print('=' * 76)
print('Stable V1.1 configuration audit')
print('=' * 76)
print(f"S-Aff scale                    : {cfg['affordance_s_scale']}")
print(f"S5A scale                      : {sms['loss_scale']}")
print(f"S5A horizons                   : {sms['horizons']}")
print(f"Outcome scale                  : {out['loss_scale']}")
print(f"Outcome evidence gate          : {out['evidence_gate_enabled']}")
print(f"Outcome gate range             : [{out['evidence_gate_low']}, {out['evidence_gate_high']}]")
print(f"Proto scale                    : {pro['loss_scale']}")
print(f"Proto EMA                      : {pro['prototype_ema']}")
print(f"Proto quantile mode            : {pro['quantile_mode']}")
print(f"Proto quantile EMA             : {pro['quantile_ema']}")
print(f"Proto boundary margin          : {pro['boundary_margin']}")
print(f"Success min support            : {pro['success_min_support']}")
print(f"Proto reliability gate         : {pro['reliability_gate_enabled']}")
print(f"Proto reliability EMA          : {pro['reliability_ema']}")
print(f"Proto reliability range        : [{pro['reliability_low']}, {pro['reliability_high']}]")
print(f"Proto reliability weights      : sem={pro['reliability_semantic_weight']}, aff={pro['reliability_affordance_weight']}")
print(f"Success reliability min gate   : {pro['success_reliability_min_gate']}")
print(f"Mask pseudo bank at gate=0     : {pro['mask_pseudo_bank_when_gate_zero']}")
print(f"Compact health log             : {cfg['compact_health_log']}")
print('=' * 76)
print('NOTE: Full heatmap, S-Aff, S5A and Outcome settings are inherited unchanged from Stable V1.')
