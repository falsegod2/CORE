"""Serial Evidence training on persistent environments from the Clean LS tree.

The environment owns task/reset/terminal semantics. This module counts actual
training actions, keeps validation out of replay, and commits resumable state.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

from .metrics import Metrics, MeanMetrics, action_summary
from .storage import Checkpoints, RunLock, atomic_json


def experiment_manifest(config):
    # Kept lazy: importing the runner does not construct a teacher or learner.
    from .configuration import experiment_manifest as make_manifest
    return make_manifest(config)


def _finite(logger, values):
    values = logger.scalar_metrics(values)
    invalid = [name for name, value in values.items() if value is None]
    if invalid:
        raise FloatingPointError('Nonfinite training metrics: ' + ', '.join(invalid[:12]))
    return values


@contextlib.contextmanager
def _evaluation_modes(method):
    """Change module modes without reseeding or restoring environment RNGs."""
    modules = []
    seen = set()
    for attribute in ('wm', 'behavior'):
        module = getattr(method, attribute, None)
        if module is not None and hasattr(module, 'modules'):
            for child in module.modules():
                if id(child) not in seen:
                    seen.add(id(child))
                    modules.append((child, child.training))
    for module, _ in modules:
        module.training = False
    try:
        yield
    finally:
        for module, training in modules:
            module.training = training


def evaluate(method, env, config, logger, step, episodes=None):
    """Evaluate serially on the same persistent Clean environment instance.

    No fixed world seed is injected and no replay methods are called. Evidence
    chooses actor.mode() through config['protocol']['eval_action'] == 'mode'.
    """
    episodes = int(config['eval_episodes'] if episodes is None else episodes)
    if episodes < 1:
        raise ValueError('Evaluation requires at least one episode')
    if config['protocol']['eval_action'] != 'mode':
        raise ValueError('Clean LS evaluation uses actor.mode(); eval_action must be mode')
    started = time.monotonic()
    rows = []
    action_counts = np.zeros(int(config['num_actions']), dtype=np.int64)
    logger.write('evaluation_start', step, split='eval', episodes=episodes,
                 evaluation_protocol='clean_ls_persistent_environment_actor_mode')
    print(f'[EVAL START] step={step} episodes={episodes}', flush=True)
    with _evaluation_modes(method):
        for episode in range(episodes):
            reset_started = time.monotonic()
            logger.write('reset_start', step, split='eval', episode=episode)
            obs = env.reset()
            logger.write('reset_ready', step, split='eval', episode=episode,
                         seconds=time.monotonic() - reset_started)
            carry = None
            score = 0.0
            success = False
            length = 0
            episode_actions = np.zeros_like(action_counts)
            first = int(config['max_episode_steps'])
            while True:
                action, carry, _ = method.act(obs, carry, training=False)
                obs = env.step(action)
                episode_actions[int(action)] += 1
                action_counts[int(action)] += 1
                length += 1
                score += float(obs['task_reward'])
                if bool(obs['success']) and not success:
                    # Keep Clean SuccessWrapper's reported step convention.
                    first = int(obs.get('first_success_step', length))
                success |= bool(obs['success'])
                if bool(obs['done']):
                    break
                if length >= config['max_episode_steps']:
                    raise RuntimeError('Clean environment crossed its task boundary without done')
            row = dict(episode=episode, return_=score, success=int(success),
                       length=length, first_success_step=first)
            rows.append(row)
            logger.write('eval_episode', step, split='eval', **row,
                         **action_summary(episode_actions, 'eval_action'))
    result = dict(eval_return=float(np.mean([row['return_'] for row in rows])),
                  eval_success=float(np.mean([row['success'] for row in rows])),
                  eval_length=float(np.mean([row['length'] for row in rows])),
                  eval_first_success_step=float(np.mean([row['first_success_step'] for row in rows])),
                  eval_episodes=episodes,
                  eval_actual_steps=sum(row['length'] for row in rows),
                  eval_seconds=time.monotonic() - started, split='eval')
    result.update(action_summary(action_counts, 'eval_action'))
    logger.write('evaluation', step, **result)
    print(f"[EVAL {step}] success={result['eval_success']:.3f} "
          f"return={result['eval_return']:.3f} seconds={result['eval_seconds']:.1f}", flush=True)
    return result


def _global_rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state()}
    try:
        import torch
    except ImportError:
        return state
    state['torch'] = torch.get_rng_state().clone()
    state['cuda'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


def _seed_everywhere(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _restore_global_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'torch' in state:
        import torch
        torch.set_rng_state(state['torch'])
        if state.get('cuda') is not None:
            if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
                raise ValueError('Checkpoint CUDA RNG topology differs from the visible devices')
            torch.cuda.set_rng_state_all(state['cuda'])


class Trainer:
    def __init__(self, config, resume=False, env_factory=None, method_factory=None):
        self.c = config
        self.dir = Path(config['logdir']).expanduser().resolve()
        self.resume = bool(resume)
        self.env_factory = env_factory
        self.method_factory = method_factory
        self.method = self.train_env = self.eval_env = None
        self.step = self.updates = self.episodes = self.evaluations = 0
        self.next_eval = int(config['prefill'])
        self.next_checkpoint = int(config['checkpoint_every'])
        self.pretrained = False
        self.last_eval = self.last_checkpoint = -1
        self.last_log = 0
        self.mean = MeanMetrics()
        self.action_counts = np.zeros(int(config['num_actions']), dtype=np.int64)
        self.prefill_action_steps = 0
        self.prefill_random = np.random.RandomState(int(config['seed']))
        self.elapsed_prior = 0.0
        self.started = 0.0
        self.next_diagnostic = int(config.get('diagnostics', {}).get('every', 20000))
        self.last_diagnostic = -1

    def _make_method(self):
        factory = self.method_factory
        if factory is None:
            from .methods.evidence_dreamer import EvidenceDreamer
            factory = EvidenceDreamer
        return factory(self.c, self.dir)

    def _make_environment(self, mode):
        factory = self.env_factory
        if factory is None:
            from .environment import CleanEvidenceEnv
            factory = CleanEvidenceEnv
        logger_mode = 'train' if mode == 'train' else 'eval'
        self.logger.write('env_initializing', self.step, split=logger_mode)
        started = time.monotonic()
        env = factory(self.c, mode=mode, run_dir=self.dir)
        if int(env.num_actions) != int(self.c['num_actions']):
            try:
                env.close()
            finally:
                raise ValueError('Clean environment action count differs from the learner configuration')
        self.logger.write('env_ready', self.step, split=logger_mode,
                          seconds=time.monotonic() - started)
        return env

    def run(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with RunLock(self.dir / '.run.lock'):
            metrics_path = self.dir / 'metrics.jsonl'
            # A missing/truncated metrics file does not make an existing run
            # new. Reject before rewriting either configuration or identity.
            existing_metadata = any((self.dir / name).exists() for name in
                                    ('config.json', 'provenance.json', 'checkpoints/latest.json'))
            committed_checkpoint = any((path / 'manifest.json').is_file()
                                       for path in (self.dir / 'checkpoints').glob('step_*'))
            nonempty_metrics = metrics_path.is_file() and metrics_path.stat().st_size > 0
            if not self.resume and (nonempty_metrics or existing_metadata or committed_checkpoint):
                raise FileExistsError(f'Run already exists: {self.dir}. Use --resume or another logdir.')
            if self.resume and not (self.dir / 'provenance.json').is_file():
                raise FileNotFoundError('Cannot resume without the original provenance.json')
            self.meta = experiment_manifest(self.c)
            if self.resume:
                stored = json.loads((self.dir / 'provenance.json').read_text())
                if stored['identity'] != self.meta['identity']:
                    raise ValueError('Resume source/config/resource identity mismatch')
            else:
                atomic_json(self.dir / 'config.json', self.c)
                atomic_json(self.dir / 'provenance.json', self.meta)
            self.logger = Metrics(metrics_path)
            self.checkpoints = Checkpoints(self.dir / 'checkpoints',
                                          self.c['keep_checkpoints'], self.c['min_free_gb'])
            self.started = time.monotonic()
            try:
                _seed_everywhere(self.c['seed'])
                learner_started = time.monotonic()
                self.logger.write('learner_initializing', self.step,
                                  method='evidence_dreamer', device=self.c['device'])
                self.method = self._make_method()
                self.logger.write('learner_ready', self.step,
                                  seconds=time.monotonic() - learner_started)
                if self.resume:
                    self.restore()
                self.logger.write('resume' if self.resume else 'launch', self.step,
                                  config=self.c, identity=self.meta['identity'],
                                  parameters=self.method.parameter_report(),
                                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                  environment_protocol='clean_ls_source_tree',
                                  validation_scope='Not paper-performance validation')
                if self.step < self.c['train_steps']:
                    self.train_env = self._make_environment('train')
                    self.eval_env = self._make_environment('eval')
                    self.loop()
                self.save()
                self.logger.write('completed', self.step, update_count=self.updates,
                                  train_wall_seconds=self.elapsed_prior + time.monotonic() - self.started)
            except BaseException as error:
                self.logger.write('failed', self.step, error_type=type(error).__name__,
                                  error=str(error), traceback=traceback.format_exc(),
                                  last_checkpoint_step=self.last_checkpoint)
                raise
            finally:
                # Do not let cleanup hide the original training failure.
                for resource in (self.eval_env, self.train_env, self.method):
                    if resource is not None:
                        try:
                            resource.close()
                        except Exception as error:
                            self.logger.write('close_error', self.step,
                                              error_type=type(error).__name__, error=str(error))

    def _evaluate(self):
        result = evaluate(self.method, self.eval_env, self.c, self.logger, self.step)
        self.last_eval = self.step
        self.evaluations += 1
        return result

    def _update_to(self, target):
        while self.updates < target:
            values = _finite(self.logger, self.method.update())
            self.mean.add(values)
            self.updates += 1

    def _diagnose(self):
        if not self.c.get('diagnostics', {}).get('enabled', False) or self.last_diagnostic == self.step:
            return
        if not self.method.ready():
            return
        started = time.monotonic()
        self.logger.write('diagnostics_start', self.step, update_count=self.updates)
        print(f'[DIAGNOSTICS START] step={self.step}', flush=True)
        try:
            result = self.method.diagnostics(self.step)
            self.logger.write('diagnostics', self.step, update_count=self.updates,
                              seconds=time.monotonic() - started, **result)
            print(f'[DIAGNOSTICS DONE] step={self.step} seconds={time.monotonic() - started:.1f}', flush=True)
        except Exception as error:
            # An optional audit must not discard an otherwise healthy run.
            # Failure remains explicit, never an invented zero-valued result.
            self.logger.write('diagnostics_failed', self.step, error_type=type(error).__name__,
                              error=str(error), traceback=traceback.format_exc(),
                              seconds=time.monotonic() - started)
            print(f'[DIAGNOSTICS FAILED] step={self.step} {type(error).__name__}: {error}', flush=True)
        self.last_diagnostic = self.step

    def _pop_train_metrics(self):
        values = self.mean.pop()
        if self.action_counts.sum():
            values.update(action_summary(self.action_counts))
            values['prefill_action_steps'] = self.prefill_action_steps
        self.action_counts.fill(0)
        self.prefill_action_steps = 0
        return values

    def loop(self):
        obs = carry = None
        score = 0.0
        success = False
        length = 0
        first_success = int(self.c['max_episode_steps'])
        while self.step < self.c['train_steps']:
            if obs is None:
                reset_started = time.monotonic()
                self.logger.write('reset_start', self.step, split='train', episode=self.episodes)
                obs = self.train_env.reset()
                self.logger.write('reset_ready', self.step, split='train', episode=self.episodes,
                                  seconds=time.monotonic() - reset_started)
                self.method.begin_episode(obs)
                carry = None
                score, success, length = 0.0, False, 0
                first_success = int(self.c['max_episode_steps'])
            forced = int(self.prefill_random.randint(self.c['num_actions'])) if self.step < self.c['prefill'] else None
            action, carry, extras = self.method.act(obs, carry, training=True, random_action=forced)
            current = self.train_env.step(action)
            self.method.record(obs, action, current, extras)
            self.step += 1
            self.action_counts[int(action)] += 1
            self.prefill_action_steps += int(forced is not None)
            length += 1
            score += float(current['task_reward'])
            if bool(current['success']) and not success:
                first_success = int(current.get('first_success_step', length))
            success |= bool(current['success'])
            if bool(current['done']):
                self.method.finish_episode()
                self.episodes += 1
                self.logger.write('train_episode', self.step, train_return=score,
                                  train_success=int(success), train_length=length,
                                  train_first_success_step=first_success,
                                  train_tail_steps=0, train_episodes=self.episodes)
                obs = carry = None
            else:
                if length >= self.c['max_episode_steps']:
                    raise RuntimeError('Clean environment crossed its task boundary without done')
                obs = current
            if self.step >= self.c['prefill'] and self.method.ready():
                if not self.pretrained:
                    # Clean LS evaluates its initialization after prefill and
                    # before its first pretraining update.
                    self._evaluate()
                    self._update_to(int(self.c['pretrain_updates']))
                    self.pretrained = True
                    self.next_eval = self.step + int(self.c['eval_every'])
                else:
                    target = int(self.c['pretrain_updates']) + int(
                        (self.step - self.c['prefill']) * self.c['train_ratio'] /
                        (self.c['batch_size'] * self.c['batch_length']))
                    self._update_to(target)
                    if self.step >= self.next_eval:
                        self._evaluate()
                        while self.next_eval <= self.step:
                            self.next_eval += int(self.c['eval_every'])
            if self.step - self.last_log >= self.c['log_every']:
                self.logger.write('train', self.step, update_count=self.updates, **self._pop_train_metrics())
                self.last_log = self.step
                print(f'[TRAIN {self.step}] updates={self.updates} episodes={self.episodes}', flush=True)
            if self.step >= self.next_diagnostic:
                self._diagnose()
                while self.next_diagnostic <= self.step:
                    self.next_diagnostic += int(self.c.get('diagnostics', {}).get('every', 20000))
            if self.c['checkpoint_every'] and self.step >= self.next_checkpoint:
                while self.next_checkpoint <= self.step:
                    self.next_checkpoint += int(self.c['checkpoint_every'])
                # The final commit follows the final evaluation and replay
                # finalization, so its RNG/schedule snapshot is the actual end.
                if self.step < self.c['train_steps']:
                    self.save()
        if obs is not None:
            # Retain valid partial replay without inventing a terminal flag.
            self.method.finish_episode()
            self.logger.write('budget_cut', self.step, partial_episode_steps=length)
        if self.method.ready() and self.last_eval != self.step:
            self._evaluate()
        self._diagnose()
        self.mean.add(_finite(self.logger, self.method.flush_metrics()))
        values = self._pop_train_metrics()
        if values:
            self.logger.write('train', self.step, update_count=self.updates, **values)

    def save(self):
        if self.step == self.last_checkpoint:
            return
        started = time.monotonic()
        self.logger.write('checkpoint_start', self.step)
        self.checkpoints.require_space()
        state = dict(method=self.method.state_dict(), step=self.step,
                     updates=self.updates, episodes=self.episodes, evaluations=self.evaluations,
                     next_eval=self.next_eval, next_checkpoint=self.next_checkpoint,
                     next_diagnostic=self.next_diagnostic, last_diagnostic=self.last_diagnostic,
                     diagnostic_settings=copy.deepcopy(self.c.get('diagnostics', {})),
                     pretrained=self.pretrained, last_eval=self.last_eval,
                     last_log=self.last_log, prefill_rng=self.prefill_random.get_state(),
                     global_rng=_global_rng_state(),
                     metric_sums=copy.deepcopy(self.mean.sums),
                     metric_counts=copy.deepcopy(self.mean.counts),
                     action_counts=self.action_counts.copy(),
                     prefill_action_steps=self.prefill_action_steps,
                     elapsed=self.elapsed_prior + time.monotonic() - self.started)
        path = self.checkpoints.save(self.step, state, self.meta['identity'])
        self.last_checkpoint = self.step
        self.logger.write('checkpoint', self.step, path=str(path), checkpoint_seconds=time.monotonic() - started)

    def restore(self):
        state = self.checkpoints.load(self.meta['identity'])
        self.method.load_state_dict(state['method'])
        for key in ('step', 'updates', 'episodes', 'evaluations', 'next_eval',
                    'next_checkpoint', 'pretrained', 'last_eval', 'last_log'):
            setattr(self, key, state[key])
        self.prefill_random.set_state(state['prefill_rng'])
        _restore_global_rng(state['global_rng'])
        self.mean.sums = copy.deepcopy(state.get('metric_sums', {}))
        self.mean.counts = copy.deepcopy(state.get('metric_counts', {}))
        self.action_counts = np.asarray(state.get('action_counts', np.zeros(
            int(self.c['num_actions']), dtype=np.int64)), dtype=np.int64).copy()
        if self.action_counts.shape != (int(self.c['num_actions']),) or (self.action_counts < 0).any():
            raise ValueError('Checkpoint action-count shape or values are invalid')
        self.prefill_action_steps = int(state.get('prefill_action_steps', 0))
        self.next_diagnostic = int(state.get('next_diagnostic', self.next_diagnostic))
        self.last_diagnostic = int(state.get('last_diagnostic', -1))
        if state.get('diagnostic_settings', {}).get('every') != self.c.get('diagnostics', {}).get('every'):
            interval = int(self.c.get('diagnostics', {}).get('every', 20000))
            self.next_diagnostic = (self.step // interval + 1) * interval
        self.elapsed_prior = float(state['elapsed'])
        self.last_checkpoint = self.step
        self.logger.write('resume_boundary', self.step,
                          note='Learner/optimizers/replay/RNG restored; Clean simulators start a new episode; old carry is not resumed.')
