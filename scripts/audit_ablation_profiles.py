#!/usr/bin/env python3
"""Static audit of the modular ablation profiles.

The important new check is that Dreamer+GenericProto keeps the single-RSSM
backend, activates only the full-latent prototype path, and does not silently
turn on S5A/Outcome/S-Aff or heatmap reconstruction.
"""

import copy
import pathlib
import types

from ruamel.yaml import YAML

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ablation

_yaml = YAML(typ="safe", pure=True)
CONFIGS = _yaml.load((ROOT / "configs.yaml").read_text())


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def resolve(profile):
    merged = {}
    recursive_update(merged, CONFIGS["defaults"])
    recursive_update(merged, CONFIGS["minedojo"])
    recursive_update(merged, CONFIGS[profile])
    cfg = types.SimpleNamespace(**merged)
    return ablation.apply_ablation_config(cfg)


# backend, saff, s5a, outcome, module_proto, source, full_proto, s_proto
EXPECTED = {
    "ab_dreamer": ("dreamer", False, False, False, False, "full", False, False),
    "ab_dreamer_proto": ("dreamer", False, False, False, True, "full", True, False),
    "ab_dual": ("dual_sz", False, False, False, False, "s", False, False),
    "ab_proto": ("dual_sz", False, False, False, True, "s", False, True),
    "ab_dual_proto": ("dual_sz", False, False, False, True, "s", False, True),
    "ab_saff": ("dual_sz", True, False, False, False, "s", False, False),
    "ab_s5a": ("dual_sz", False, True, False, False, "s", False, False),
    "ab_outcome": ("dual_sz", False, False, True, False, "s", False, False),
    "ab_saff_s5a": ("dual_sz", True, True, False, False, "s", False, False),
    "ab_saff_s5a_outcome": ("dual_sz", True, True, True, False, "s", False, False),
    "ab_saff_s5a_proto": ("dual_sz", True, True, False, True, "s", False, True),
    "ab_full": ("dual_sz", True, True, True, True, "s", False, True),
}

for profile, expected in EXPECTED.items():
    cfg = resolve(profile)
    sms = cfg.s_multi_step_consistency
    actual = (
        cfg.model_backend,
        bool(cfg.module_saff),
        bool(cfg.module_s5a),
        bool(sms["outcome"]["enabled"]),
        bool(cfg.module_proto),
        cfg.proto_latent_source,
        bool(cfg.full_latent_proto_enabled),
        bool(sms["prototype_utility"]["enabled"]),
    )
    assert actual == expected, (profile, actual, expected)

    if not cfg.module_saff:
        assert cfg.decoder["cnn_keys"] == "^image$", (profile, cfg.decoder)
        assert cfg.full_heatmap_scale == 0.0
    else:
        assert "heatmap" in cfg.decoder["cnn_keys"]

# The exact requested ablation must not secretly enable S5A or Outcome.
cfg = resolve("ab_dreamer_proto")
sms = cfg.s_multi_step_consistency
assert cfg.model_backend == "dreamer"
assert cfg.module_dual_sz is False
assert cfg.full_latent_proto_enabled is True
assert cfg.proto_latent_source == "full"
assert sms["enabled"] is False, "single-stream Proto must not activate S5A machinery"
assert float(sms["loss_scale"]) == 0.0
assert sms["outcome"]["enabled"] is False
assert sms["prototype_utility"]["enabled"] is False, "S-branch Proto must stay off"
assert cfg.decoder["cnn_keys"] == "^image$"
assert cfg.use_heatmap_aux is True, "affordance teacher still needs the U-Net heatmap"

# Dual+Proto uses the original S-branch route, with rollout machinery but no S5A loss.
cfg = resolve("ab_dual_proto")
sms = cfg.s_multi_step_consistency
assert cfg.model_backend == "dual_sz"
assert sms["enabled"] is True
assert float(sms["loss_scale"]) == 0.0
assert sms["prototype_utility"]["enabled"] is True
assert cfg.full_latent_proto_enabled is False

print("PASS: Dreamer+GenericProto and all core ablation profiles are structurally clean.")
