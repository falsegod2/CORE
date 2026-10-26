"""Serial scheduling and complete checkpoints around original LS_Imagine.

The original tools.simulate owns zoom labels and tail interactions. Step counts
include every training environment action, including those tail interactions.
"""
from __future__ import annotations
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import tools
from expr import LS_Imagine
from parallel import Damy
from clean_evidence.metrics import Metrics
from clean_evidence.storage import RunLock, Checkpoints, atomic_json
from clean_evidence.runner import _global_rng_state, _restore_global_rng
from .configuration import namespace, experiment_manifest


class _RunEvents:
    """Keep runtime compatibility while making metrics.jsonl self-contained.

    Native LS scalar rows remain unique. Evaluation completion events carry
    their repeated scalar summary under ``result`` rather than repeating flat
    ``eval_success`` fields which older plotting scripts count as evaluations.
    """
    def __init__(self, directory):
        self.primary = Metrics(directory / 'metrics.jsonl')
        self.compatibility = Metrics(directory / 'runtime.jsonl')

    def write(self, event, step, **values):
        self.primary.write(event, step, **values)
        self.compatibility.write(event, step, **values)


class _ObservedLogger:
    """Observe the original logger without changing LS learning or scalars.

    Extra evaluation timing/count scalars are written in the original, single
    evaluation row. The wrapped native logger retains TensorBoard/video behavior.
    """
    def __init__(self, logger):
        object.__setattr__(self, '_native', logger)
        object.__setattr__(self, '_pending', {})
        object.__setattr__(self, '_evaluation_started', None)
        object.__setattr__(self, '_evaluation_steps', None)
        object.__setattr__(self, 'last_evaluation', None)

    def __getattr__(self, name):
        return getattr(self._native, name)

    def __setattr__(self, name, value):
        if name in ('_native', '_pending', '_evaluation_started',
                    '_evaluation_steps', 'last_evaluation'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._native, name, value)

    def scalar(self, name, value):
        self._pending[name] = float(value)
        self._native.scalar(name, value)

    def begin_evaluation(self, started, steps):
        self.last_evaluation = None
        self._evaluation_started, self._evaluation_steps = started, steps

    def end_evaluation(self):
        self._evaluation_started = self._evaluation_steps = None

    def write(self, *args, **kwargs):
        if self._evaluation_started is not None and 'eval_success' in self._pending:
            self.scalar('eval_actual_steps', int(self._evaluation_steps()))
            self.scalar('eval_seconds', time.monotonic() - self._evaluation_started)
            self.last_evaluation = {key: value for key, value in self._pending.items()
                                    if key.startswith('eval_')}
        result = self._native.write(*args, **kwargs)
        self._pending.clear()
        return result


def _device_metadata(device):
    """Report visibility rather than guessing the physical CUDA index."""
    result = dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  logical_device=str(device), device_name=None)
    if str(device).startswith('cuda') and torch.cuda.is_available():
        result['device_name'] = torch.cuda.get_device_name(torch.device(device))
    return result


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _optimizers(agent):
    return dict(model=agent._wm._model_opt, actor=agent._task_behavior._actor_opt,
                value=agent._task_behavior._value_opt)


class Trainer:
    def __init__(self, config, resume=False, env_factory=None):
        self.config, self.c, self.resume = config, namespace(config), resume
        self.dir = Path(config['logdir'])
        self.env_factory = env_factory
        self.agent = self.logger = None
        self.envs = []
        self.train_eps, self.eval_eps = OrderedDict(), OrderedDict()
        # Original tools.sample_episodes used an independent seed=0 generator.
        self.train_rng, self.eval_rng = np.random.RandomState(0), np.random.RandomState(0)
        self.next_eval = self.c.prefill
        self.next_checkpoint = self.c.checkpoint_every
        self.last_eval = self.last_checkpoint = -1
        self.step = 0
        self.state = None

    def run(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with RunLock(self.dir / '.run.lock'):
            existing = any((self.dir / n).exists() for n in
                           ('config.json', 'provenance.json', 'metrics.jsonl', 'runtime.jsonl', 'checkpoints/latest.json'))
            existing |= any((path / 'manifest.json').is_file()
                            for path in (self.dir / 'checkpoints').glob('step_*'))
            existing |= any((self.dir / 'train_eps').glob('*.npz'))
            if existing and not self.resume:
                raise FileExistsError(f'Run already exists: {self.dir}. Use --resume or another --logdir.')
            self.meta = experiment_manifest(self.config)
            if self.resume:
                saved = json.loads((self.dir / 'provenance.json').read_text())
                if saved['identity'] != self.meta['identity']:
                    raise ValueError('LS resume source/config/resource identity mismatch')
            else:
                atomic_json(self.dir / 'config.json', self.config)
                atomic_json(self.dir / 'provenance.json', self.meta)
            self.events = _RunEvents(self.dir)
            self.checkpoints = Checkpoints(self.dir / 'checkpoints', self.config['keep_checkpoints'],
                                           self.config['min_free_gb'])
            for path in (self.c.traindir, self.c.evaldir):
                path.mkdir(parents=True, exist_ok=True)
            tools.set_seed_everywhere(self.c.seed)
            if self.c.deterministic_run:
                tools.enable_deterministic_run()
            self.logger = _ObservedLogger(tools.Logger(self.c, self.dir, 0))
            try:
                self._initialize()
                self.events.write('resume' if self.resume else 'launch', self.step,
                                  method='ls_imagine', config=self.config,
                                  runtime=_device_metadata(self.c.device),
                                  count_scope='all train environment actions including LS label tails')
                self._loop()
                self.save()
                self.events.write('completed', self.step)
            except BaseException as error:
                self.events.write('failed', self.step, error_type=type(error).__name__,
                                  error=str(error), traceback=traceback.format_exc(),
                                  last_checkpoint_step=self.last_checkpoint)
                raise
            finally:
                for env in self.envs:
                    try:
                        env.close()
                    except Exception as error:
                        self.events.write('close_error', self.step, error=str(error))
                if self.logger:
                    self.logger.finish()
                    if hasattr(self.logger, '_writer'):
                        self.logger._writer.close()

    def _initialize(self):
        if self.env_factory is None:
            from .environment import make_environment
            factory = make_environment
        else:
            factory = self.env_factory
        for mode in ('train', 'eval'):
            self.events.write('env_initializing', self.step, split=mode)
            started = time.monotonic()
            env = factory(self.config, mode=mode, run_dir=self.dir)
            self.envs.append(Damy(env))
            self.events.write('env_ready', self.step, split=mode, seconds=time.monotonic() - started)
        self.train_score = tools.ScoreStorage(self.c.episode_max_steps)
        self.eval_score = tools.ScoreStorage(self.c.episode_max_steps)
        acts = self.envs[0].action_space
        self.c.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]
        if self.c.num_actions != 12:
            raise ValueError('Expected original Clean 12-action environment')
        # Match original Clean order: collect random prefill before constructing
        # the LS learner, so initialization does not consume prefill RNG draws.
        if not self.resume:
            self._prefill()
        data = tools.from_generator(tools.sample_episodes(self.train_eps, self.c.batch_length,
                                                         rng=self.train_rng), self.c.batch_size)
        self.eval_dataset = tools.from_generator(tools.sample_episodes(self.eval_eps, self.c.batch_length,
                                                                       rng=self.eval_rng), self.c.batch_size)
        self.agent = LS_Imagine(self.envs[0].observation_space, acts, self.c, self.logger, data).to(self.c.device)
        self.agent.requires_grad_(False)
        if self.resume:
            self.restore()

    def _simulate(self, agent, steps=0, evaluation=False, prefill=False):
        return tools.simulate(
            agent, [self.envs[1 if evaluation else 0]],
            self.eval_eps if evaluation else self.train_eps,
            self.c.evaldir if evaluation else self.c.traindir, self.logger,
            self.eval_score if evaluation else self.train_score,
            self.c.episode_max_steps, self.c.discount,
            is_eval=evaluation, limit=None if evaluation else self.c.dataset_size,
            steps=steps, episodes=self.c.eval_episode_num if evaluation else 0,
            state=None if evaluation else self.state, is_training=not (evaluation or prefill))

    def _evaluate(self):
        started = time.monotonic()
        self.events.write('evaluation_start', self.step, episodes=self.c.eval_episode_num)
        # envs=1 is validated by configuration. Each call results in exactly
        # one real environment action; the legacy simulator owns all semantics.
        actual_steps = 0
        def evaluation_agent(*args, **kwargs):
            nonlocal actual_steps
            result = self.agent(*args, training=False, **kwargs)
            actual_steps += 1
            return result
        self.logger.begin_evaluation(started, lambda: actual_steps)
        try:
            simulation = self._simulate(evaluation_agent, evaluation=True)
        finally:
            self.logger.end_evaluation()
        result = copy.deepcopy(self.logger.last_evaluation)
        if result is None or int(result['eval_episodes']) != self.c.eval_episode_num:
            raise RuntimeError('LS evaluation did not produce its complete native scalar summary')
        if int(simulation[0]) != actual_steps:
            raise RuntimeError('LS evaluation action count disagrees with native simulation')
        if self.c.video_pred_log:
            video = self.agent._wm.video_pred(next(self.eval_dataset))
            self.logger.video('eval_openl', video.detach().cpu().numpy())
            self.logger.write()
        self.last_eval = self.step
        self.events.write('evaluation', self.step, split='eval', result=result,
                          eval_actual_steps=actual_steps,
                          eval_seconds=time.monotonic() - started,
                          timing_scope='complete evaluation including video logging',
                          scalar_timing_scope='environment and policy before final logger flush')

    def _prefill(self):
        if self.step < self.c.prefill:
            random_actor = tools.OneHotDist(torch.zeros(1, self.c.num_actions))
            def random_agent(obs, reset, state):
                action = random_actor.sample()
                return {'action': action, 'logprob': random_actor.log_prob(action)}, None
            count = min(self.c.prefill, self.c.steps) - self.step
            self.state = self._simulate(random_agent, steps=count, prefill=True)
            self.step += count
            self.logger.step = self.step
            self.events.write('prefill_done', self.step)

    def _loop(self):
        while self.step < self.c.steps:
            if self.step >= self.next_eval:
                self._evaluate()
                while self.next_eval <= self.step:
                    self.next_eval += self.c.eval_every
            boundary = min(self.c.steps, self.next_eval, self.next_checkpoint)
            if boundary > self.step:
                self.state = self._simulate(self.agent, steps=boundary - self.step)
                self.step = int(self.agent._step)
                self.events.write('train_progress', self.step, updates=self.agent._update_count)
            if self.step >= self.next_checkpoint:
                while self.next_checkpoint <= self.step:
                    self.next_checkpoint += self.c.checkpoint_every
                if self.step < self.c.steps:
                    self.save()
        if self.last_eval != self.step:
            self._evaluate()

    def save(self):
        if self.last_checkpoint == self.step:
            return
        started = time.monotonic()
        schedules = {name: copy.deepcopy(getattr(self.agent, name).__dict__) for name in
                     ('_should_train', '_should_pretrain', '_should_log', '_should_reset')}
        opts = {name: dict(optimizer=_cpu(opt._opt.state_dict()), scaler=opt._scaler.state_dict())
                for name, opt in _optimizers(self.agent).items()}
        state = dict(step=self.step, model=_cpu(self.agent.state_dict()), optimizers=opts,
                     train_eps=self.train_eps, eval_eps=self.eval_eps,
                     train_rng=self.train_rng.get_state(), eval_rng=self.eval_rng.get_state(),
                     global_rng=_global_rng_state(), schedules=schedules,
                     updates=self.agent._update_count,
                     behavior_updates=getattr(self.agent._task_behavior, '_updates', 0),
                     jump_prob=self.agent._task_behavior.jump_prob,
                     pending_metrics=copy.deepcopy(self.agent._metrics),
                     next_eval=self.next_eval, next_checkpoint=self.next_checkpoint, last_eval=self.last_eval)
        path = self.checkpoints.save(self.step, state, self.meta['identity'])
        self.last_checkpoint = self.step
        self.events.write('checkpoint', self.step, path=str(path),
                          checkpoint_seconds=time.monotonic() - started)

    def restore(self):
        state = self.checkpoints.load(self.meta['identity'])
        self.agent.load_state_dict(state['model'])
        for name, opt in _optimizers(self.agent).items():
            opt._opt.load_state_dict(state['optimizers'][name]['optimizer'])
            opt._scaler.load_state_dict(state['optimizers'][name]['scaler'])
        self.train_eps.update(state['train_eps'])
        self.eval_eps.update(state['eval_eps'])
        self.train_rng.set_state(state['train_rng'])
        self.eval_rng.set_state(state['eval_rng'])
        for name, values in state['schedules'].items():
            getattr(self.agent, name).__dict__.update(values)
        for key in ('step', 'next_eval', 'next_checkpoint', 'last_eval'):
            setattr(self, key, state[key])
        self.agent._step = self.logger.step = self.step
        self.agent._update_count = state['updates']
        self.agent._task_behavior._updates = state['behavior_updates']
        self.agent._task_behavior.jump_prob = state['jump_prob']
        self.agent._metrics = state['pending_metrics']
        self.last_checkpoint = self.step
        _restore_global_rng(state['global_rng'])
        self.events.write('resume_boundary', self.step,
                          note='Model/optimizers/scalers/replay/sampler/RNG/schedules restored; simulator and unfinished LS label tails restart; incomplete zoom labels remain invalid.')
