from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)
cfgs = yaml.load((Path(__file__).parent / "configs.yaml").read_text())


def merge(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v

cfg = {}
for name in ["defaults", "minedojo", "stable_v1", "stable_v1_1", "stable_v1_2"]:
    merge(cfg, cfgs[name])

proto = cfg["s_multi_step_consistency"]["prototype_utility"]
outcome = cfg["s_multi_step_consistency"]["outcome"]

print("Stable V1.2 audit")
print("=" * 62)
print("S-Aff scale                 :", cfg["affordance_s_scale"])
print("S5A scale                   :", cfg["s_multi_step_consistency"]["loss_scale"])
print("Outcome scale/gate          :", outcome["loss_scale"], outcome["evidence_gate_enabled"])
print("Proto scale/reliability     :", proto["loss_scale"], proto["reliability_gate_enabled"])
print("Proto gradient guard        :", proto["gradient_guard_enabled"])
print("Guard interval              :", proto["gradient_guard_interval"])
print("Guard conflict range        :", [proto["gradient_guard_conflict_low"], proto["gradient_guard_conflict_high"]])
print("Guard min conflict gate     :", proto["gradient_guard_min_conflict_gate"])
print("Guard max aux/base norm     :", proto["gradient_guard_max_norm_ratio"])
print("Guard minimum total gate    :", proto["gradient_guard_min_total_gate"])
print("Success replay              :", cfg["success_replay_enabled"])
print("Success replay ratio        :", cfg["success_replay_ratio"])
print("=" * 62)
print("Stable V1.2 intentionally leaves Full Heatmap / S-Aff / S5A / Outcome / actor / critic unchanged.")
