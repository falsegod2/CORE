"""Original LS wrappers with explicit resource paths; no Evidence wrapper."""
import copy
from pathlib import Path


def make_environment(config, mode, run_dir):
    from envs.tasks import get_specs
    from envs.tasks.minedojo import make_minedojo
    from envs import wrappers
    clean = config['clean']
    task_id, task, sim = get_specs(config['task'].removeprefix('minedojo_'),
                                   target_item=clean['target_item'])
    task, sim = copy.deepcopy(task), copy.deepcopy(sim)
    root = Path(config['assets_root'])
    concentration = task['concentration_specs']
    concentration.update(max_steps=task['terminal_specs']['max_steps'],
                         gaussian_reward_weight=clean['gaussian_reward_weight'],
                         gaussian_sigma_weight=clean['gaussian_sigma_weight'])
    conditions = task['success_specs'].get('all') or task['success_specs'].get('any') or {}
    task['clip_specs']['target_object'] = conditions.get('item', {}).get('type', clean['target_item'])
    unet = Path(concentration['unet_checkpoint_dir'])
    if not unet.is_absolute():
        unet = root / unet
    task['ls_runtime_specs'] = dict(clip_checkpoint=str(root / 'weights/mineclip_attn.pth'),
                                   unet_checkpoint_dir=str(unet), device=config['env_device'])
    output = Path(run_dir) / ('environment_' + mode)
    output.mkdir(parents=True, exist_ok=True)
    task['log_dir'] = str(output)
    env = make_minedojo(task_id, task, sim)
    return wrappers.UUID(wrappers.SelectAction(wrappers.OneHotAction(env), key='action'))
