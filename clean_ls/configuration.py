"""Pure configuration; importing this module does not initialize CUDA."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from clean_evidence.configuration import (
    PROJECT_ROOT, absolute, default_assets_root, merge, read_yaml,
    resource_paths, set_value, task_specs,
)


def build_config(task='minedojo_harvest_log_in_plains', seed=0, stage='formal',
                 logdir=None, assets_root=None, overrides=()):
    native = read_yaml(PROJECT_ROOT / 'configs.yaml')
    clean = merge(native['defaults'], native['minedojo'])
    _, specs = task_specs(task)
    clean.update(task=task, seed=int(seed), compile=False,
                 checkpoint_every=20000, episode_max_steps=int(specs['terminal_specs']['max_steps']))
    for key in ('steps', 'dataset_size', 'prefill', 'pretrain'):
        clean[key] = int(clean[key])
    config = dict(method='ls_imagine', stage=stage, task=task, seed=int(seed),
                  assets_root=absolute(assets_root or default_assets_root()),
                  logdir=absolute(logdir or PROJECT_ROOT / 'logdir/ls_imagine' / stage / task / f'seed_{seed}'),
                  device='cuda:0', env_device='cuda:0', keep_checkpoints=2,
                  min_free_gb=8., clean=clean)
    if stage == 'smoke':
        clean.update(steps=5000, prefill=64, pretrain=1, batch_size=2, batch_length=8,
                     eval_every=2500, eval_episode_num=1, log_every=100,
                     checkpoint_every=2500, dyn_deter=32, dyn_hidden=16,
                     dyn_stoch=4, dyn_discrete=4, units=32, imag_horizon=5,
                     max_imag_sequences_num=32, video_pred_log=False)
        for name in ('encoder', 'decoder'):
            clean[name].update(cnn_depth=4, mlp_layers=1, mlp_units=32)
        for name in ('actor', 'critic', 'reward_head', 'end_head', 'jump_head',
                     'concentration_score_head', 'intrinsic_head', 'jumping_steps_head',
                     'accumulated_reward_head'):
            clean[name]['layers'] = 1
    elif stage != 'formal':
        raise ValueError('stage must be formal or smoke')
    for expression in overrides:
        set_value(config, expression)
    config['clean'].update(task=config['task'], seed=config['seed'], device=config['device'])
    config['logdir'], config['assets_root'] = absolute(config['logdir']), absolute(config['assets_root'])
    validate(config)
    return config


def validate(config):
    c = config['clean']
    if config['method'] != 'ls_imagine':
        raise ValueError('This entry point runs ls_imagine')
    if c['envs'] != 1 or c['action_repeat'] != 1 or c['parallel']:
        raise ValueError('Managed Clean LS requires envs=1, action_repeat=1, parallel=false')
    if c['compile'] or c['expl_behavior'] != 'greedy':
        raise ValueError('Managed LS checkpoints support compile=false and original greedy exploration')
    if c['offline_traindir'] or c['offline_evaldir'] or c['traindir'] or c['evaldir']:
        raise ValueError('Managed LS replay is stored inside its exact run directory')
    for key in ('steps', 'prefill', 'batch_size', 'batch_length', 'eval_every',
                'eval_episode_num', 'log_every', 'checkpoint_every', 'dataset_size'):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'clean.{key} must be a positive integer')
    if c['steps'] < c['prefill'] or c['batch_length'] < 3 or c['train_ratio'] <= 0:
        raise ValueError('Need steps >= prefill, batch_length >= 3 and positive train_ratio')
    if isinstance(c['train_ratio'], bool) or not isinstance(c['train_ratio'], (int, float)) or not math.isfinite(c['train_ratio']):
        raise ValueError('clean.train_ratio must be finite and positive')
    if isinstance(c['pretrain'], bool) or not isinstance(c['pretrain'], int) or c['pretrain'] < 0:
        raise ValueError('clean.pretrain must be a nonnegative integer')
    if isinstance(config['keep_checkpoints'], bool) or not isinstance(config['keep_checkpoints'], int) or config['keep_checkpoints'] < 1:
        raise ValueError('keep_checkpoints must be a positive integer')
    if isinstance(config['min_free_gb'], bool) or not isinstance(config['min_free_gb'], (int, float)) or not math.isfinite(config['min_free_gb']) or config['min_free_gb'] < 0:
        raise ValueError('min_free_gb must be finite and nonnegative')
    _, specs = task_specs(config['task'])
    if c['episode_max_steps'] != int(specs['terminal_specs']['max_steps']):
        raise ValueError('Episode limit must come from Clean task_specs.yaml')


def namespace(config):
    c = copy.deepcopy(config['clean'])
    root = Path(config['logdir'])
    c.update(logdir=root, traindir=root / 'train_eps', evaldir=root / 'eval_eps',
             results_dir=str(root / 'environment'), num_actions=12)
    return SimpleNamespace(**c)


def experiment_manifest(config):
    from clean_evidence.storage import sha256
    paths = [p for folder in ('envs', 'clean_ls', 'clean_evidence/runtime')
             for p in (PROJECT_ROOT / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.yaml', '.png')]
    paths += [PROJECT_ROOT / n for n in ('expr.py', 'models.py', 'networks.py', 'tools.py',
                                        'exploration.py', 'parallel.py', 'configs.yaml',
                                        'run_ls_imagine.py', 'clean_evidence/configuration.py',
                                        'clean_evidence/storage.py', 'clean_evidence/runner.py',
                                        'clean_evidence/metrics.py', 'clean_evidence/ablations.py')]
    sources = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sorted(paths)}
    assets = {k: dict(path=str(p), sha256=sha256(p)) for k, p in resource_paths(config).items()}
    clean = {k: v for k, v in config['clean'].items()
             if k not in ('steps', 'device', 'logdir', 'traindir', 'evaldir', 'results_dir')}
    payload = dict(method=config['method'], stage=config['stage'], clean=clean, sources=sources,
                   assets={k: v['sha256'] for k, v in assets.items()})
    return dict(identity=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                sources=sources, assets=assets,
                note='Original LS zoom/tail learning; simulator is restarted on resume')
