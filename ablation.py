"""Runtime configuration for controlled DreamerV3 module ablations.

The public knobs are deliberately flat so they can be changed from the command
line. The function maps them onto the nested config expected by the current
Dual-S/Z implementation.
"""


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def apply_ablation_config(config):
    dual = _as_bool(getattr(config, "module_dual_sz", False))
    saff = _as_bool(getattr(config, "module_saff", False))
    s5a = _as_bool(getattr(config, "module_s5a", False))
    outcome = _as_bool(getattr(config, "module_outcome", False))
    proto = _as_bool(getattr(config, "module_proto", False))
    inverse = _as_bool(getattr(config, "module_inverse", False))
    z_adv = _as_bool(getattr(config, "module_z_adv", False))

    dependent = {
        "S-Aff": saff,
        "S5A": s5a,
        "BalancedOutcome": outcome,
        "GenericProto": proto,
        "Inverse": inverse,
        "ZAdv": z_adv,
    }
    if not dual and any(dependent.values()):
        active = [name for name, enabled in dependent.items() if enabled]
        raise ValueError(
            "Modules %s require module_dual_sz=True because they operate on "
            "the controllable S branch." % active
        )

    # Shared reward shaping knob. 1.0 reproduces the existing experiments;
    # 0.0 gives environment-reward-only actor/critic training.
    if not hasattr(config, "intrinsic_reward_scale"):
        config.intrinsic_reward_scale = 1.0

    # Map flat ablation knobs onto the current implementation.
    config.inverse_loss_scale = (
        float(getattr(config, "inverse_scale", 1.0)) if inverse else 0.0
    )
    config.z_action_adv_scale = (
        float(getattr(config, "z_adv_scale", 1.0)) if z_adv else 0.0
    )
    config.affordance_s_scale = (
        float(getattr(config, "saff_scale", 1.0)) if saff else 0.0
    )
    config.full_heatmap_scale = (
        float(getattr(config, "full_heatmap_scale", 1.0)) if saff else 0.0
    )

    sms = dict(getattr(config, "s_multi_step_consistency", {}) or {})
    outcome_cfg = dict(sms.get("outcome", {}) or {})
    proto_cfg = dict(sms.get("prototype_utility", {}) or {})

    # Outcome and Proto reuse the S5A rollout machinery even when the S5A
    # consistency loss itself is zero. This makes true "module-only" ablations
    # possible without silently adding the S5A loss.
    sms["enabled"] = bool(s5a or outcome or proto)
    sms["loss_scale"] = (
        float(getattr(config, "s5a_scale", 0.05)) if s5a else 0.0
    )

    outcome_cfg["enabled"] = outcome
    outcome_cfg["loss_scale"] = (
        float(getattr(config, "outcome_scale", 0.01)) if outcome else 0.0
    )

    proto_cfg["enabled"] = proto
    proto_cfg["loss_scale"] = (
        float(getattr(config, "proto_scale", 0.01)) if proto else 0.0
    )
    proto_cfg["label_mode"] = str(
        getattr(config, "proto_label_mode", proto_cfg.get("label_mode", "task_evidence"))
    )

    sms["outcome"] = outcome_cfg
    sms["prototype_utility"] = proto_cfg
    config.s_multi_step_consistency = sms

    # Decoder supervision must be controlled too. Otherwise turning S-Aff off
    # while leaving `heatmap` in the decoder would still train a hidden
    # full-latent heatmap auxiliary loss, invalidating the ablation.
    decoder = dict(config.decoder)
    decoder["cnn_keys"] = "^(image|heatmap)$" if saff else "^image$"
    config.decoder = decoder

    # Heatmaps are generated only when an active module actually needs them.
    # Generic Proto needs them when its affordance teacher has non-zero weight.
    proto_needs_heatmap = proto and float(proto_cfg.get("affordance_weight", 0.30)) > 0.0
    config.use_heatmap_aux = bool(saff or proto_needs_heatmap)

    config.model_backend = "dual_sz" if dual else "dreamer"

    return config


def summary_lines(config):
    sms = getattr(config, "s_multi_step_consistency", {}) or {}
    out = sms.get("outcome", {}) or {}
    proto = sms.get("prototype_utility", {}) or {}
    return [
        "[ABLATION] backend               = %s" % getattr(config, "model_backend", "?"),
        "[ABLATION] Dual S/Z              = %s" % bool(getattr(config, "module_dual_sz", False)),
        "[ABLATION] S-Aff                 = %s (scale=%g, full_heatmap=%g)" % (
            bool(getattr(config, "module_saff", False)),
            float(getattr(config, "affordance_s_scale", 0.0)),
            float(getattr(config, "full_heatmap_scale", 0.0)),
        ),
        "[ABLATION] S5A                   = %s (scale=%g)" % (
            bool(getattr(config, "module_s5a", False)),
            float(sms.get("loss_scale", 0.0)),
        ),
        "[ABLATION] Balanced Outcome      = %s (scale=%g)" % (
            bool(out.get("enabled", False)), float(out.get("loss_scale", 0.0))
        ),
        "[ABLATION] Generic Proto         = %s (scale=%g, mode=%s)" % (
            bool(proto.get("enabled", False)),
            float(proto.get("loss_scale", 0.0)),
            proto.get("label_mode", "task_evidence"),
        ),
        "[ABLATION] Inverse               = %s (scale=%g)" % (
            bool(getattr(config, "module_inverse", False)),
            float(getattr(config, "inverse_loss_scale", 0.0)),
        ),
        "[ABLATION] Z action adversary    = %s (scale=%g)" % (
            bool(getattr(config, "module_z_adv", False)),
            float(getattr(config, "z_action_adv_scale", 0.0)),
        ),
        "[ABLATION] intrinsic reward      = scale=%g" % float(
            getattr(config, "intrinsic_reward_scale", 1.0)
        ),
        "[ABLATION] heatmap teacher pipe  = %s" % bool(
            getattr(config, "use_heatmap_aux", False)
        ),
    ]


def print_summary(config):
    print("\n" + "=" * 72)
    print("DreamerV3 modular ablation configuration")
    print("=" * 72)
    for line in summary_lines(config):
        print(line)
    print("=" * 72 + "\n")
