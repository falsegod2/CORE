from omegaconf import OmegaConf
from envs.tasks.minedojo import make_minedojo

CUTOM_TASK_SPECS = OmegaConf.to_container(OmegaConf.load("envs/tasks/task_specs.yaml"))

def get_specs(task, **kwargs):
    # Route simulator-level reproducibility options into sim_specs rather than
    # forwarding them to unrelated task wrappers.
    sim_seed = kwargs.pop("sim_seed", None)
    world_seed = kwargs.pop("world_seed", None)
    force_hard_reset = bool(kwargs.pop("force_hard_reset", False))
    clip_score_quantum = float(kwargs.pop("clip_score_quantum", 0.0))
    clip_improvement_eps = float(kwargs.pop("clip_improvement_eps", 0.0))
    affordance_score_quantum = float(kwargs.pop("affordance_score_quantum", 0.0))
    affordance_map_quantum = float(kwargs.pop("affordance_map_quantum", 0.0))

    # Get task data and task id
    if task in CUTOM_TASK_SPECS:
        yaml_specs = CUTOM_TASK_SPECS[task].copy()
        task_id = yaml_specs.pop("task_id", task)
        assert "sim" in yaml_specs, "task_specs.yaml must define sim attribute"
    else:
        yaml_specs = dict()
        task_id = task

    # Get minedojo specs
    sim_specs = yaml_specs.pop("sim_specs", dict())

    if sim_seed is not None:
        sim_specs["seed"] = int(sim_seed)
    if world_seed is not None and str(world_seed):
        sim_specs["world_seed"] = str(world_seed)

    # Get our task specs
    task_specs = dict(
        clip=False,
        fake_clip=False,
        fake_dreamer=False,
        subgoals=False,
    )
    task_specs.update(**yaml_specs)
    task_specs.update(**kwargs)
    if force_hard_reset:
        # Semi-fast reset preserves previous world edits and amplifies tiny
        # trajectory differences. Hard reset is slower but more reproducible.
        task_specs["fast_reset"] = None
    if task_specs.get("clip_specs") is not None:
        task_specs["clip_specs"]["score_quantum"] = clip_score_quantum
        task_specs["clip_specs"]["improvement_eps"] = clip_improvement_eps
    if task_specs.get("concentration_specs") is not None:
        task_specs["concentration_specs"]["score_quantum"] = affordance_score_quantum
        task_specs["concentration_specs"]["map_quantum"] = affordance_map_quantum
        task_specs["concentration_specs"]["improvement_eps"] = clip_improvement_eps
    assert not (task_specs["clip"] and task_specs["fake_clip"]), "Can only use one reward shaper"

    return task_id, task_specs, sim_specs

def make(task: str, **kwargs):
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict
    env = make_minedojo(task_id, task_specs, sim_specs)

    return env
