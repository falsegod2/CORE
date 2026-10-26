"""Evidence model recipe over the uploaded Clean LS task and runtime defaults."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from ruamel.yaml import YAML

from .ablations import apply_ablation, validate_ablation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        return YAML(typ='safe').load(stream) or {}


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def absolute(value):
    return str(Path(os.path.expandvars(str(value))).expanduser().resolve())


def default_assets_root():
    if os.environ.get('CORE_ASSETS'):
        return absolute(os.environ['CORE_ASSETS'])
    cjj = Path.home() / 'CJJ'
    candidates = [PROJECT_ROOT, cjj / 'LS_Imagine_Clean_Evidence_20261020',
                  cjj / 'DreamerV3_Clean_PixelOnly_BinaryReward/CORE-CORE_DREAMERV3_MINEDOJO',
                  cjj / 'CORE', cjj]
    for path in candidates:
        if (path / 'weights/mineclip_attn.pth').is_file():
            return absolute(path)
    return absolute(cjj / 'DreamerV3_Clean_PixelOnly_BinaryReward/CORE-CORE_DREAMERV3_MINEDOJO')


def set_value(config, expression):
    if '=' not in expression:
        raise ValueError('Use --set path=value')
    path, raw = expression.split('=', 1)
    if path == 'ablation':
        raise ValueError('Use --ablation to select a named control, not --set ablation=...')
    parts = path.split('.')
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise KeyError(f'Unknown config parent: {path}')
        current = current[part]
    if parts[-1] not in current:
        raise KeyError(f'Unknown Evidence setting: {path}')
    current[parts[-1]] = YAML(typ='safe').load(raw)


def task_specs(task):
    if not task.startswith('minedojo_'):
        raise ValueError('Use a Clean task name such as minedojo_harvest_log_in_plains')
    name = task[len('minedojo_'):]
    specs = read_yaml(PROJECT_ROOT / 'envs/tasks/task_specs.yaml')
    if name not in specs:
        raise ValueError(f'No task definition in the uploaded Clean LS: {name}')
    return name, copy.deepcopy(specs[name])


def build_config(task='minedojo_harvest_log_in_plains', seed=0, stage='formal',
                 logdir=None, assets_root=None, overrides=(), ablation='full'):
    if stage not in ('formal', 'smoke'):
        raise ValueError('stage must be formal or smoke')
    native = read_yaml(PROJECT_ROOT / 'configs.yaml')
    clean = merge(native['defaults'], native['minedojo'])
    name, specs = task_specs(task)
    recipe = read_yaml(PACKAGE_ROOT / 'configs/evidence.yaml')
    config = dict(
        method='evidence_dreamer', stage=stage, task=task, seed=int(seed),
        environment='uploaded_clean_minedojo', device=clean['device'], env_device=clean['device'],
        assets_root=absolute(assets_root or default_assets_root()),
        logdir=absolute(logdir or PROJECT_ROOT / 'logdir/evidence_clean' / ablation / stage / task / f'seed_{seed}'),
        train_steps=int(clean['steps']), prefill=int(clean['prefill']),
        pretrain_updates=int(clean['pretrain']), batch_size=int(clean['batch_size']),
        batch_length=int(clean['batch_length']), train_ratio=float(clean['train_ratio']),
        replay_capacity=int(clean['dataset_size']), eval_every=int(clean['eval_every']),
        eval_episodes=int(clean['eval_episode_num']), log_every=int(clean['log_every']),
        checkpoint_every=20000, keep_checkpoints=2, min_free_gb=8.0,
        diagnostics=dict(enabled=True, every=20000, batch_size=2, gradients=True),
        num_actions=12, max_episode_steps=int(specs['terminal_specs']['max_steps']),
        needs_heatmap=True, semantic_teacher_only=True,
        protocol=dict(name='uploaded_clean_ls_environment', eval_action='mode',
                      reward='native_task_plus_clean_mineclip_record_high',
                      reset='original_clean_semifast', fast_resets=specs.get('fast_reset'),
                      task_reward_specs=copy.deepcopy(specs.get('reward_specs'))),
        dreamer=dict(dynamics='evidence', both_action=True, match_shared_initialization=False,
                     loss_plugins=[], overrides=copy.deepcopy(recipe['model'])),
        evidence=copy.deepcopy(recipe['evidence']), clean=clean,
    )
    if stage == 'smoke':
        config.update(train_steps=5000, prefill=64, pretrain_updates=1, batch_size=2,
                      batch_length=8, eval_every=2500, eval_episodes=1,
                      log_every=100, checkpoint_every=2500)
        model = config['dreamer']['overrides']
        model.update(dyn_deter=32, dyn_hidden=16, dyn_stoch=4, dyn_discrete=4,
                     units=32, imag_horizon=5)
        for section in ('encoder', 'decoder'):
            model[section].update(cnn_depth=4, mlp_layers=1, mlp_units=32)
        for section in ('actor', 'critic', 'reward_head', 'cont_head'):
            model[section]['layers'] = 1
        config['evidence'].update(memory_observations=2, memory_dim=16, attention_heads=2,
                                  writer_depth=4, burnin=4, horizons=[1, 2, 4],
                                  horizon_weights=[1., 1., .5], starts_per_sequence=2)
        config['diagnostics']['every'] = 2500
    apply_ablation(config, ablation)
    for expression in overrides:
        set_value(config, expression)
    config['assets_root'] = absolute(config['assets_root'])
    config['logdir'] = absolute(config['logdir'])
    config['clean'].update(task=config['task'], seed=config['seed'], device=config['env_device'])
    validate(config)
    return config


def validate(config):
    name, specs = task_specs(config['task'])
    if config['environment'] != 'uploaded_clean_minedojo':
        raise ValueError('Evidence uses the uploaded Clean MineDojo environment; no fallback adapter')
    if int(config['clean'].get('envs', 1)) != 1 or int(config['clean'].get('action_repeat', 1)) != 1:
        raise ValueError('This Evidence bridge supports Clean envs=1 and action_repeat=1')
    if config['num_actions'] != 12:
        raise ValueError('Preserve the 12 real actions defined by Clean LSImagineWrapper')
    if config['max_episode_steps'] != int(specs['terminal_specs']['max_steps']):
        raise ValueError('Task episode limit comes from Clean task_specs.yaml; do not override it here')
    if config['protocol']['eval_action'] != 'mode':
        raise ValueError('The supplied Clean evaluation protocol uses actor.mode()')
    if not config['semantic_teacher_only']:
        raise ValueError('Enabled Evidence semantic targets must remain training-only')
    validate_ablation(config)
    if config['protocol']['name'] != 'uploaded_clean_ls_environment' or config['protocol']['reset'] != 'original_clean_semifast':
        raise ValueError('Retain the uploaded Clean environment protocol')
    if config['protocol']['fast_resets'] != specs.get('fast_reset'):
        raise ValueError('Reset schedule must come from the uploaded task specification')
    for key in ('train_steps', 'prefill', 'batch_size', 'batch_length', 'eval_every',
                'eval_episodes', 'log_every', 'checkpoint_every', 'keep_checkpoints', 'replay_capacity'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['train_steps'] < config['prefill'] or config['batch_length'] < 3:
        raise ValueError('Need train_steps >= prefill and batch_length >= 3')
    if isinstance(config['pretrain_updates'], bool) or not isinstance(config['pretrain_updates'], int) or config['pretrain_updates'] < 0:
        raise ValueError('pretrain_updates must be a nonnegative integer')
    if not math.isfinite(config['train_ratio']) or config['train_ratio'] <= 0:
        raise ValueError('train_ratio must be finite and positive')
    options = config['evidence']
    diagnostic = config['diagnostics']
    for key in ('enabled', 'gradients'):
        if not isinstance(diagnostic[key], bool):
            raise ValueError(f'diagnostics.{key} must be a boolean')
    for key in ('every', 'batch_size'):
        if isinstance(diagnostic[key], bool) or not isinstance(diagnostic[key], int) or diagnostic[key] < 1:
            raise ValueError(f'diagnostics.{key} must be a positive integer')
    if diagnostic['batch_size'] > 16:
        raise ValueError('diagnostics.batch_size must be <= 16; diagnostic probes are bounded')
    for key in ('memory_observations', 'memory_dim', 'attention_heads', 'writer_depth',
                'heatmap_size', 'starts_per_sequence', 'mc_samples'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f'evidence.{key} must be a positive integer')
    if isinstance(options['burnin'], bool) or not isinstance(options['burnin'], int):
        raise ValueError('Evidence burnin must be an integer')
    if options['burnin'] < options['memory_observations'] or options['memory_dim'] % options['attention_heads']:
        raise ValueError('Evidence burnin must cover memory; memory_dim must divide attention_heads')
    if options['tokens_per_observation'] != 4 or options['heatmap_size'] < 2 or options['heatmap_size'] % 2:
        raise ValueError('Evidence uses four quadrant tokens and an even heatmap_size >= 2')
    for key in ('memory_enabled', 'freeze_rc_reward_head'):
        if not isinstance(options[key], bool):
            raise ValueError(f'evidence.{key} must be boolean')
    if options['read_mode'] not in ('each_step', 'start_only', 'none'):
        raise ValueError('Unsupported evidence.read_mode')
    for key in ('local_heat_weight', 'posterior_heat_weight', 'fc_weight', 'fc_hidden_weight',
                'fc_heat_weight', 'rc_weight', 'task_reward_weight', 'clip_reward_weight', 'clip_scale'):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'evidence.{key} must be finite and nonnegative')
    if options['clip_scale'] != 1.0:
        raise ValueError('The Clean MineCLIP signal is retained at scale 1')
    horizons, weights = options['horizons'], options['horizon_weights']
    if not horizons or len(set(horizons)) != len(horizons) or len(weights) != len(horizons):
        raise ValueError('Unique positive horizons must match horizon_weights')
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in horizons):
        raise ValueError('Evidence horizons must be positive integers')
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Invalid horizon weight')
    model = config['dreamer']['overrides']
    if config['dreamer']['dynamics'] != 'evidence' or not config['dreamer']['both_action']:
        raise ValueError('Evidence requires its joint action-conditioned S/Z dynamics')
    for key in ('dyn_deter', 'dyn_hidden', 'dyn_stoch', 'dyn_discrete', 'units', 'imag_horizon'):
        if isinstance(model[key], bool) or not isinstance(model[key], int) or model[key] < 1:
            raise ValueError(f'dreamer.overrides.{key} must be positive integer')
    if any(model[k] % 2 for k in ('dyn_deter', 'dyn_hidden', 'dyn_stoch')) or model['imag_horizon'] < 3:
        raise ValueError('Joint branches require even dimensions and imag_horizon >= 3')
    for section in ('encoder', 'decoder'):
        if model[section]['cnn_keys'] not in ('image', '^image$') or model[section]['mlp_keys'] != '$^':
            raise ValueError('Evidence encodes/reconstructs RGB only; teacher heatmaps are targets')


def resource_paths(config):
    name, specs = task_specs(config['task'])
    root = Path(config['assets_root'])
    paths = {'mineclip': root / 'weights/mineclip_attn.pth'}
    if not config.get('needs_heatmap', True):
        return paths
    unet = specs.get('concentration_specs', {}).get('unet_checkpoint_dir')
    if not unet:
        raise ValueError('This task has no semantic teacher checkpoint directory in Clean task specs')
    paths['unet'] = root / unet / 'swin_unet_checkpoint.pth'
    return paths


def experiment_manifest(config):
    from .storage import sha256
    paths = [p for folder in (PROJECT_ROOT / 'envs', PACKAGE_ROOT)
             for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.png')
             and '__pycache__' not in p.parts]
    paths += [PROJECT_ROOT / name for name in ('configs.yaml', 'run_evidence.py')]
    sources = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sorted(paths)}
    assets = {}
    for key, path in resource_paths(config).items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {key} resource: {path}')
        assets[key] = {'path': str(path), 'sha256': sha256(path)}
    ignored = {'train_steps', 'logdir', 'assets_root', 'device', 'env_device', 'min_free_gb', 'keep_checkpoints', 'diagnostics'}
    semantic_config = {key: value for key, value in config.items() if key not in ignored}
    # Toggle/cadence/gradient measurement are operational. Retain the bank
    # size in identity so a resumed audit cannot silently change its sample
    # design while the first nonzero bank is still pending.
    semantic_config['diagnostics'] = {'batch_size': config.get('diagnostics', {}).get('batch_size', 2)}
    # Device and output path inside the copied Clean settings are operational.
    semantic_config['clean'] = {key: value for key, value in config['clean'].items()
                                if key not in ('device', 'logdir', 'traindir', 'evaldir', 'results_dir')}
    payload = {'config': semantic_config, 'sources': sources,
               'assets': {key: value['sha256'] for key, value in assets.items()}}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {'identity': identity, 'sources': sources, 'assets': assets,
            'environment_source': 'uploaded Clean LS task/action/reset/HUD wrappers',
            'not_claimed': 'bit-exact Minecraft continuation or control-performance validation'}
