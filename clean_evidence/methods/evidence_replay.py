"""Incoming-action replay with real-history burn-in for EvidenceDreamer.

One sample contains a contiguous portion of exactly one real episode. Padding
is explicit and never masquerades as an environmental reset or observation.
Teacher labels are retained as targets; the RGB writer never receives them.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np

from ..rng import seed_for


class EvidenceReplay:
    def __init__(self, config, options=None):
        self.config = config
        self.options = copy.deepcopy(config.get('evidence', {}) if options is None else options)
        self.burnin = self.options.get('burnin', 32)
        memory_observations = self.options.get('memory_observations', 16)
        if isinstance(self.burnin, bool) or not isinstance(self.burnin, int) or self.burnin < 0:
            raise ValueError('evidence.burnin must be a nonnegative integer')
        if isinstance(memory_observations, bool) or not isinstance(memory_observations, int) or memory_observations < 1:
            raise ValueError('evidence.memory_observations must be a positive integer')
        if self.burnin < memory_observations:
            raise ValueError('evidence.burnin must cover memory_observations to reconstruct the FIFO')
        self.episodes = OrderedDict()
        self.active = None
        self.counter = 0
        self.rng = np.random.RandomState(seed_for(config['seed'], 'evidence_replay'))

    def _row(self, obs, action=None):
        image = np.asarray(obs['image'])
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError('Evidence replay requires uint8 RGB of shape (64,64,3)')
        if action is not None and not 0 <= int(action) < self.config['num_actions']:
            raise ValueError(f'Action outside shared discrete action space: {action}')
        first = bool(obs['is_first'])
        if first != (action is None):
            raise ValueError('Replay reset and incoming action disagree with the true is_first flag')
        incoming = np.zeros(self.config['num_actions'], np.float32)
        if action is not None:
            incoming[int(action)] = 1.0
        # The Clean environment provides its MineCLIP record-high intrinsic
        # before optional LS concentration shaping. Keep that native label; do
        # not replace it with the old WM adapter's delta-score reward.
        task_reward = 0.0 if first else float(obs['task_reward'])
        clip_reward = 0.0 if first else float(obs.get('clip_reward', obs.get('intrinsic', 0.0)))
        if not np.isfinite(task_reward) or not np.isfinite(clip_reward):
            raise FloatingPointError('Nonfinite task or MineCLIP reward in evidence replay')
        heatmap_valid = bool(obs.get('heatmap_valid', 'heatmap' in obs))
        heatmap = np.asarray(obs.get('heatmap', np.zeros((64, 64), np.uint8)))
        if heatmap.shape == (64, 64, 1):
            heatmap = heatmap[..., 0]
        if heatmap.shape != (64, 64) or heatmap.dtype != np.uint8:
            raise ValueError('Teacher heatmap must be uint8 with shape (64,64) or (64,64,1)')
        if heatmap_valid and 'heatmap' not in obs:
            raise ValueError('heatmap_valid cannot be true without a teacher target')
        # Preserve Clean's explicit discount, including time-limit conventions.
        discount = np.asarray(obs.get('discount', float(not bool(obs['is_terminal']))))
        if discount.size != 1:
            raise ValueError('Clean discount must be one finite scalar')
        discount = float(discount.reshape(()))
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError('Clean discount must be finite and within [0,1]')
        # Keep mixed policy reward for audits, plus independent target labels.
        reward = task_reward + clip_reward
        return {
            'image': image.copy(),
            'action': incoming,
            'reward': np.float32(reward),
            'task_reward': np.float32(task_reward),
            'clip_reward': np.float32(clip_reward),
            'heatmap': heatmap.copy(),
            'heatmap_valid': np.bool_(heatmap_valid),
            'discount': np.float32(discount),
            'is_first': np.bool_(first),
            'is_last': np.bool_(obs['is_last']),
            'is_terminal': np.bool_(obs['is_terminal']),
        }

    def begin(self, obs):
        if self.active is not None:
            raise RuntimeError('Finish the preceding replay episode before starting another')
        self.counter += 1
        self.active = str(self.counter)
        self.episodes[self.active] = [self._row(obs)]

    def add(self, action, obs):
        if self.active is None:
            raise RuntimeError('begin_episode must precede recording a transition')
        rows = self.episodes[self.active]
        if rows[-1]['is_last']:
            raise RuntimeError('Cannot record a transition after the actual episode end')
        rows.append(self._row(obs, action))

    def finish(self):
        if self.active is not None:
            rows = self.episodes[self.active]
            self.episodes[self.active] = {key: np.stack([row[key] for row in rows])
                                         for key in rows[0]}
        self.active = None
        # A run budget cut does not imply an environmental terminal or last row.
        while self.steps() > self.config['replay_capacity'] and len(self.episodes) > 1:
            self.episodes.popitem(last=False)

    @staticmethod
    def _length(episode):
        return len(episode['reward']) if isinstance(episode, dict) else len(episode)

    def steps(self):
        return sum(max(0, self._length(episode) - 1) for episode in self.episodes.values())

    def ready(self):
        return any(self._length(episode) > 1 for episode in self.episodes.values())

    @staticmethod
    def _slice(episode, start, stop):
        if isinstance(episode, dict):
            return {key: value[start:stop].copy() for key, value in episode.items()}
        return {key: np.stack([row[key] for row in episode[start:stop]])
                for key in episode[0]}

    def sample(self):
        if not self.ready():
            raise RuntimeError('Evidence replay has no executed transition')
        episodes = [episode for episode in self.episodes.values()
                    if self._length(episode) > 1]
        # Uniform executed-transition starts across the available real episodes.
        counts = np.asarray([self._length(episode) - 1 for episode in episodes], np.float64)
        probabilities = counts / counts.sum()
        learning_length = self.config['batch_length']
        total_length = self.burnin + learning_length
        batch = []
        for _ in range(self.config['batch_size']):
            episode = episodes[int(self.rng.choice(len(episodes), p=probabilities))]
            length = self._length(episode)
            learning_start = int(self.rng.randint(1, length))
            history_start = max(0, learning_start - self.burnin)
            stop = min(length, learning_start + learning_length)
            segment = self._slice(episode, history_start, stop)
            left_padding = self.burnin - (learning_start - history_start)
            real_length = stop - history_start
            sampled = {}
            for key, value in segment.items():
                output = np.zeros((total_length,) + value.shape[1:], dtype=value.dtype)
                output[left_padding:left_padding + real_length] = value
                sampled[key] = output
            sampled['sequence_valid'] = np.zeros(total_length, np.bool_)
            sampled['sequence_valid'][left_padding:left_padding + real_length] = True
            sampled['loss_mask'] = np.zeros(total_length, np.bool_)
            sampled['loss_mask'][self.burnin:self.burnin + stop - learning_start] = True
            batch.append(sampled)
        return {key: np.stack([sampled[key] for sampled in batch]) for key in batch[0]}

    def state_dict(self):
        return {
            'episodes': self._snapshot_episodes(self.episodes),
            'counter': self.counter,
            'rng': self.rng.get_state(),
            'burnin': self.burnin,
            'learning_length': self.config['batch_length'],
        }

    def load_state_dict(self, state):
        if int(state.get('burnin', self.burnin)) != self.burnin:
            raise ValueError('Evidence replay checkpoint burn-in differs')
        if int(state.get('learning_length', self.config['batch_length'])) != self.config['batch_length']:
            raise ValueError('Evidence replay checkpoint learning length differs')
        self.episodes = self._snapshot_episodes(state['episodes'])
        self.counter = int(state['counter'])
        self.rng.set_state(state['rng'])
        # Resuming the learner starts a new simulator episode. Old partial rows
        # remain valid replay; an old simulator carry is never resumed silently.
        self.active = None

    @staticmethod
    def _snapshot_episodes(episodes):
        # Rows are immutable and sample copies slices. Avoid doubling millions
        # of RGB/teacher targets at each checkpoint, as in the native adapter.
        return OrderedDict((key, dict(episode) if isinstance(episode, dict)
                            else [dict(row) for row in episode])
                           for key, episode in episodes.items())
