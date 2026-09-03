# coding=utf-8
"""Lazy backend dispatcher for the modular DreamerV3 ablation baseline."""


def _backend_name(config):
    return "modular" if bool(getattr(config, "module_dual_sz", False)) else "dreamer"


def WorldModel(obs_space, act_space, step, config):
    backend = _backend_name(config)
    if backend == "dreamer":
        import models_dreamer as backend_models
    else:
        import models_modular as backend_models
    model = backend_models.WorldModel(obs_space, act_space, step, config)
    model._ablation_backend = backend
    return model


def ImagBehavior(config, world_model):
    backend = getattr(world_model, "_ablation_backend", _backend_name(config))
    if backend == "dreamer":
        import models_dreamer as backend_models
    else:
        import models_modular as backend_models
    return backend_models.ImagBehavior(config, world_model)
