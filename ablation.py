"""Runtime configuration for controlled DreamerV3 module ablations.

This version additionally supports a *single-stream* Dreamer + Generic Proto
ablation.  Generic Proto can therefore supervise either:
  - ``s``: the controllable S branch of Dual S/Z; or
  - ``full``: the standard single-stream Dreamer RSSM full latent.

The switch is explicit so the experiment can separate the contribution of the
Dual split from the contribution of the prototype objective itself.
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

    source = str(getattr(config, "proto_latent_source", "auto")).lower()
    if source not in {"auto", "s", "full"}:
        raise ValueError(
            "proto_latent_source must be one of auto/s/full, got %r" % source
        )
    if source == "auto":
        source = "s" if dual else "full"
    config.proto_latent_source = source

    # These modules are mathematically defined on the controllable S branch and
    # therefore require Dual S/Z. Generic Proto is the deliberate exception:
    # its controlled ablation may use the ordinary Dreamer full latent.
    dependent = {
        "S-Aff": saff,
        "S5A": s5a,
        "BalancedOutcome": outcome,
        "Inverse": inverse,
        "ZAdv": z_adv,
    }
    if not dual and any(dependent.values()):
        active = [name for name, enabled in dependent.items() if enabled]
        raise ValueError(
            "Modules %s require module_dual_sz=True because they operate on "
            "the controllable S branch." % active
        )
    if proto and source == "s" and not dual:
        raise ValueError(
            "proto_latent_source='s' requires module_dual_sz=True. Use 'full' "
            "for Dreamer + Generic Proto."
        )
    if proto and source == "full" and dual:
        raise ValueError(
            "For a controlled Dual S/Z experiment, Generic Proto must use "
            "proto_latent_source='s'. Use module_dual_sz=False for the full "
            "single-stream Dreamer latent ablation."
        )

    if not hasattr(config, "intrinsic_reward_scale"):
        config.intrinsic_reward_scale = 1.0

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

    # The Dual/S version may reuse S5A rollout machinery without enabling the
    # S5A cosine objective. The single-stream Dreamer+Proto version uses its own
    # native RSSM img_step path in models_dreamer.py and does not activate S5A.
    s_proto = bool(proto and source == "s")
    full_proto = bool(proto and source == "full")
    sms["enabled"] = bool(dual and (s5a or outcome or s_proto))
    sms["loss_scale"] = (
        float(getattr(config, "s5a_scale", 0.05)) if s5a else 0.0
    )

    outcome_cfg["enabled"] = outcome
    outcome_cfg["loss_scale"] = (
        float(getattr(config, "outcome_scale", 0.01)) if outcome else 0.0
    )

    proto_cfg["enabled"] = s_proto
    proto_cfg["loss_scale"] = (
        float(getattr(config, "proto_scale", 0.01)) if s_proto else 0.0
    )
    proto_cfg["label_mode"] = str(
        getattr(config, "proto_label_mode", proto_cfg.get("label_mode", "task_evidence"))
    )

    sms["outcome"] = outcome_cfg
    sms["prototype_utility"] = proto_cfg
    config.s_multi_step_consistency = sms

    # Separate runtime switch used only by the standard single-RSSM backend.
    config.full_latent_proto_enabled = full_proto
    config.full_latent_proto_scale = (
        float(getattr(config, "proto_scale", 0.01)) if full_proto else 0.0
    )

    # Turning S-Aff off must also remove the hidden full-latent heatmap decoder
    # target. Generic Proto may still consume heatmap as a teacher signal.
    decoder = dict(config.decoder)
    decoder["cnn_keys"] = "^(image|heatmap)$" if saff else "^image$"
    config.decoder = decoder

    proto_needs_heatmap = proto and float(proto_cfg.get("affordance_weight", 0.30)) > 0.0
    config.use_heatmap_aux = bool(saff or proto_needs_heatmap)
    config.model_backend = "dual_sz" if dual else "dreamer"
    return config


def summary_lines(config):
    sms = getattr(config, "s_multi_step_consistency", {}) or {}
    out = sms.get("outcome", {}) or {}
    s_proto = sms.get("prototype_utility", {}) or {}
    proto = bool(getattr(config, "module_proto", False))
    proto_scale = float(getattr(config, "proto_scale", 0.0)) if proto else 0.0
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
        "[ABLATION] Generic Proto         = %s (source=%s, scale=%g, mode=%s)" % (
            proto,
            getattr(config, "proto_latent_source", "?"),
            proto_scale,
            getattr(config, "proto_label_mode", "task_evidence"),
        ),
        "[ABLATION] Full-latent Proto     = %s" % bool(
            getattr(config, "full_latent_proto_enabled", False)
        ),
        "[ABLATION] S-branch Proto        = %s" % bool(s_proto.get("enabled", False)),
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
