"""Immutable replay probes; never a training sampler or held-out benchmark.

Each run retains an ordinary fixed sample and a separately identified,
reward-stratified sample. They remain fixed after capture, including on resume.
No simulator calls, global RNG draws, or training replay RNG draws occur here.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np

from .metrics import MeanMetrics
from .rng import isolated_rng, seed_for


class FixedReplayProbes:
    VERSION = 1

    def __init__(self, config):
        self.config = config
        self.banks = {}

    @staticmethod
    def _digest(batch):
        digest = hashlib.sha256()
        for key, value in sorted(batch.items()):
            value = np.ascontiguousarray(value)
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _segment(replay, episode, learning_start):
        burnin = replay.burnin
        learning_length = replay.config['batch_length']
        length = replay._length(episode)
        history_start = max(0, learning_start - burnin)
        stop = min(length, learning_start + learning_length)
        source = replay._slice(episode, history_start, stop)
        left = burnin - (learning_start - history_start)
        count = stop - history_start
        total = burnin + learning_length
        sample = {}
        for key, value in source.items():
            out = np.zeros((total,) + value.shape[1:], dtype=value.dtype)
            out[left:left + count] = value
            sample[key] = out
        sample['sequence_valid'] = np.zeros(total, dtype=np.bool_)
        sample['sequence_valid'][left:left + count] = True
        sample['loss_mask'] = np.zeros(total, dtype=np.bool_)
        sample['loss_mask'][burnin:burnin + stop - learning_start] = True
        return sample

    def capture_missing(self, replay, step):
        """Capture each bank once. An absent nonzero bank remains explicitly pending."""
        if len(self.banks) == 2:
            return
        eligible = [(key, ep) for key, ep in replay.episodes.items()
                    if replay._length(ep) >= 3]
        if not eligible:
            return
        size = self.config['diagnostics']['batch_size']
        if 'natural' not in self.banks:
            rng = np.random.RandomState(seed_for(self.config['seed'], 'probe', 'natural'))
            # Like training, weight episodes by executed transitions. Exclude
            # the final row as a start, because a diagnostic needs a future.
            counts = np.asarray([replay._length(ep) - 2 for _, ep in eligible], np.float64)
            samples, locations = [], []
            for _ in range(size):
                key, episode = eligible[int(rng.choice(len(eligible), p=counts / counts.sum()))]
                start = int(rng.randint(1, replay._length(episode) - 1))
                samples.append(self._segment(replay, episode, start))
                locations.append({'episode': key, 'learning_start': start})
            self._store('natural', samples, locations, step,
                        'transition-weighted fixed sample; not held-out data')
        if 'reward_nonzero' not in self.banks:
            candidates = []
            for key, episode in eligible:
                rewards = (np.asarray(episode['task_reward']) if isinstance(episode, dict)
                           else np.asarray([row['task_reward'] for row in episode]))
                # A reward at t is predicted from earlier posterior states;
                # t>=2 leaves a non-reset, executed-observation start at t-1.
                for t in np.flatnonzero(np.abs(rewards) > 1e-6):
                    spans = [h for h, weight in zip(self.config['evidence']['horizons'],
                                                   self.config['evidence']['horizon_weights'])
                             if weight > 0 and h <= int(t) - 1 and h < self.config['batch_length']]
                    if spans:
                        candidates.append((key, episode, int(t), max(spans)))
            if candidates:
                rng = np.random.RandomState(seed_for(self.config['seed'], 'probe', 'reward_nonzero'))
                choice = rng.choice(len(candidates), size=min(size, len(candidates)), replace=False)
                samples, locations = [], []
                for index in choice:
                    key, episode, reward_time, horizon = candidates[int(index)]
                    # Align the reward with an actual scored horizon, including
                    # short episodes whose length falls between H values.
                    # Deterministic model starts include this first valid row.
                    start = reward_time - horizon
                    samples.append(self._segment(replay, episode, start))
                    locations.append({'episode': key, 'learning_start': start,
                                      'reward_time': reward_time})
                self._store('reward_nonzero', samples, locations, step,
                            'reward-stratified fixed sample; not prevalence or unbiased evaluation')

    def _store(self, name, samples, locations, step, scope):
        batch = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        self.banks[name] = dict(batch=batch, fingerprint=self._digest(batch),
                                captured_at_step=int(step), locations=locations, scope=scope)

    def state_dict(self):
        return {'version': self.VERSION, 'banks': copy.deepcopy(self.banks)}

    def load_state_dict(self, state):
        if state.get('version') != self.VERSION:
            raise ValueError('Unsupported fixed diagnostic probe version')
        banks = copy.deepcopy(state['banks'])
        if set(banks) - {'natural', 'reward_nonzero'}:
            raise ValueError('Unknown fixed diagnostic bank')
        for bank in banks.values():
            if self._digest(bank['batch']) != bank['fingerprint']:
                raise ValueError('Fixed diagnostic bank fingerprint mismatch')
        self.banks = banks

    def evaluate(self, method, step):
        """Read-only forward/autograd probes with training randomness restored."""
        self.capture_missing(method.replay, step)
        settings = self.config['diagnostics']
        results = {}
        modules = list(method.wm.modules())
        modes = [(module, module.training) for module in modules]
        try:
            method.wm.eval()
            for name, bank in self.banks.items():
                started = time.monotonic()
                # Stable stream per bank across checkpoints; isolated even if
                # diagnostic code raises. Replay sampling state is never used.
                with isolated_rng(seed_for(self.config['seed'], 'fixed_probe_forward', name)):
                    metrics = method.wm.diagnostic_metrics(bank['batch'], gradients=settings['gradients'])
                values = {}
                for key, value in metrics.items():
                    value = float(value.detach().cpu().item()) if hasattr(value, 'detach') else float(value)
                    if not math.isfinite(value):
                        raise FloatingPointError(f'Nonfinite diagnostic metric: {name}.{key}')
                    values[key] = value
                aggregate = MeanMetrics()
                aggregate.add(values)
                values = aggregate.pop()
                for key, value in list(values.items()):
                    if key.endswith('_mse'):
                        values[key[:-4] + '_rmse'] = math.sqrt(max(value, 0.0)) if value is not None else None
                results[name] = dict(fingerprint=bank['fingerprint'],
                                     captured_at_step=bank['captured_at_step'],
                                     batch_sequences=len(bank['locations']),
                                     locations=copy.deepcopy(bank['locations']),
                                     scope=bank['scope'], seconds=time.monotonic() - started,
                                     metrics=values)
        finally:
            for module, mode in modes:
                module.training = mode
        return dict(banks=results,
                    pending_banks=[name for name in ('natural', 'reward_nonzero') if name not in self.banks],
                    protocol='fixed_recorded_replay_v1', gradients=bool(settings['gradients']),
                    weights_changed=False, updates_performed=0)
