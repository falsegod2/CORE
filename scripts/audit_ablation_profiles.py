#!/usr/bin/env python3
import copy
import pathlib
import types

from ruamel.yaml import YAML

import ablation

ROOT = pathlib.Path(__file__).resolve().parents[1]
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


EXPECTED = {
    "ab_dreamer": ("dreamer", False, False, False, False),
    "ab_dual": ("dual_sz", False, False, False, False),
    "ab_saff": ("dual_sz", True, False, False, False),
    "ab_s5a": ("dual_sz", False, True, False, False),
    "ab_outcome": ("dual_sz", False, False, True, False),
    "ab_proto": ("dual_sz", False, False, False, True),
    "ab_saff_s5a": ("dual_sz", True, True, False, False),
    "ab_saff_s5a_outcome": ("dual_sz", True, True, True, False),
    "ab_saff_s5a_proto": ("dual_sz", True, True, False, True),
    "ab_full": ("dual_sz", True, True, True, True),
}

for profile, expected in EXPECTED.items():
    cfg = resolve(profile)
    sms = cfg.s_multi_step_consistency
    actual = (
        cfg.model_backend,
        bool(cfg.module_saff),
        bool(cfg.module_s5a),
        bool(sms["outcome"]["enabled"]),
        bool(sms["prototype_utility"]["enabled"]),
    )
    assert actual == expected, (profile, actual, expected)

    # Hidden heatmap decoder loss must not remain when S-Aff is off.
    if not cfg.module_saff:
        assert cfg.decoder["cnn_keys"] == "^image$", (profile, cfg.decoder)
        assert cfg.full_heatmap_scale == 0.0
    else:
        assert "heatmap" in cfg.decoder["cnn_keys"]

    # Proto/Outcome may activate rollout machinery but must not silently add S5A.
    if profile in {"ab_outcome", "ab_proto"}:
        assert sms["enabled"] is True
        assert float(sms["loss_scale"]) == 0.0

print("PASS: all core ablation profiles resolve to the intended modules.")
