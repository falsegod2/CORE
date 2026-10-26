"""Bridge the supplied Clean LS MineDojo chain to Evidence replay packets.

Physics, task reward, native terminal flags, actions and semifast reset come
from the Clean source. No world seed bank or simulator reconstruction is added.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


class CleanEvidenceEnv:
    def __init__(self, config, mode='train', run_dir=None):
        if mode not in ('train', 'eval'):
            raise ValueError('CleanEvidenceEnv mode must be train or eval')
        task = config['task']
        prefix = 'minedojo_'
        if not task.startswith(prefix):
            raise ValueError('Evidence currently supports the supplied minedojo_ Clean tasks')
        task_name = task[len(prefix):]
        self.config = config
        self.mode = mode
        self._teacher_enabled = mode == 'train' and config.get('needs_heatmap', True)
        self._closed = False
        self._steps = 0
        self._first_success = None
        root = Path(config['assets_root']).expanduser().resolve()
        clean = config.get('clean') or {}
        log_dir = Path(run_dir) if run_dir is not None else Path(clean.get('results_dir', 'results')) / clean.get('name', 'evidence')
        log_dir = (log_dir / ('environment_' + mode)).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        # These are precisely the source task factory and outer action wrappers
        # used by expr.make_env. The entry point runs from the project root,
        # matching its relative task YAML and screenshot HUD-mask resources.
        from envs.tasks import get_specs
        from envs.tasks.minedojo import make_minedojo
        from envs import wrappers

        task_id, task_specs, sim_specs = get_specs(task_name, target_item=clean.get('target_item', 'log'))
        task_specs, sim_specs = copy.deepcopy(task_specs), copy.deepcopy(sim_specs)
        self.max_episode_steps = int(task_specs['terminal_specs']['max_steps'])
        self.episode_max_steps = self.max_episode_steps
        concentration = task_specs.get('concentration_specs')
        if self._teacher_enabled and not concentration:
            raise ValueError('Evidence training requires Clean task concentration_specs')
        if concentration is not None:
            concentration['max_steps'] = self.max_episode_steps
            concentration['gaussian_reward_weight'] = clean.get('gaussian_reward_weight', 1.0)
            concentration['gaussian_sigma_weight'] = clean.get('gaussian_sigma_weight', 0.5)
        success = task_specs.get('success_specs') or {}
        conditions = success.get('all') or success.get('any') or {}
        if task_specs.get('clip_specs') is not None:
            task_specs['clip_specs']['target_object'] = (conditions.get('item') or {}).get('type', clean.get('target_item', 'log'))

        # Test task names keep their own Clean fast_reset and resource paths.
        # Do not rewrite eval to test_<task>: Clean expr.make_env does not do so.
        unet_relative = (concentration or {}).get('unet_checkpoint_dir')
        if unet_relative is None:
            unet_relative = 'affordance_map/finetune_unet/finetune_checkpoints/' + task_name.removeprefix('test_')
        unet = Path(unet_relative).expanduser()
        if not unet.is_absolute():
            unet = root / unet
        task_specs['evidence_specs'] = {
            'training': mode == 'train',
            'semantic_teacher': self._teacher_enabled,
            'device': config.get('env_device') or config['device'],
            'clip_checkpoint': str(root / 'weights/mineclip_attn.pth'),
            'unet_checkpoint_dir': str(unet),
        }
        task_specs['log_dir'] = str(log_dir)
        env = make_minedojo(task_id, task_specs, sim_specs)
        env = wrappers.OneHotAction(env)
        env = wrappers.SelectAction(env, key='action')
        self.env = wrappers.UUID(env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_actions = int(self.action_space.shape[0])
        if self.num_actions != 12:
            self.close()
            raise ValueError('The supplied Clean MineDojo action wrapper must expose its original 12 actions')

    def _packet(self, observation, reward=0.0, done=False, info=None):
        info = info or {}
        packet = dict(observation)
        success = bool(info.get('success', False))
        if success and self._first_success is None:
            self._first_success = int(info.get('first_success_step', self._steps))
        packet.update(
            task_reward=float(reward), reward=float(reward),
            clip_reward=float(info.get('clip_score', 0.0)),
            heatmap_valid=self._teacher_enabled, done=bool(done), success=success,
            episode_step=self._steps,
            first_success_step=int(info.get('first_success_step', self._first_success if self._first_success is not None else self.max_episode_steps)),
            discount=float(info.get('discount', 1.0 - float(done))),
            episode_id=self.env.id,
        )
        # is_first/is_last/is_terminal remain exactly those emitted by Clean's
        # LSImagineWrapper. A successful task is not reclassified as terminal.
        return packet

    def reset(self):
        if self._closed:
            raise RuntimeError('Environment is closed')
        self._steps = 0
        self._first_success = None
        return self._packet(self.env.reset())

    def step(self, action):
        if self._closed:
            raise RuntimeError('Environment is closed')
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.num_actions:
            raise ValueError('Action must be an integer in the Clean 12-action space')
        onehot = np.zeros(self.num_actions, dtype=np.float32)
        onehot[int(action)] = 1.0
        observation, reward, done, info = self.env.step({'action': onehot})
        self._steps += 1
        return self._packet(observation, reward, done, info)

    def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self, 'env'):
                self.env.close()
