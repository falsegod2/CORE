"""Actual LS learner on deterministic test-only physics, including resume."""
import copy
import hashlib
import json
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(1)

# Gym 0.21 expects the pre-Python-3.12 entry_points mapping API.
import importlib.metadata as metadata
if not hasattr(metadata.entry_points(), 'get'):
    _entry_points = metadata.entry_points
    metadata.entry_points = lambda **kw: _entry_points(**kw) if kw else {}

from clean_ls import runner
from clean_ls.configuration import build_config


class DummyWriter:
    def close(self):
        pass


class FakeLogger:
    def __init__(self, config, logdir, step):
        self.step, self.values, self._writer = step, {}, DummyWriter()
    def scalar(self, name, value):
        assert np.isfinite(value), name
        self.values[name] = float(value)
    def video(self, *args):
        pass
    def write(self, *args, **kwargs):
        pass
    def finish(self):
        pass


class FakeEnv:
    def __init__(self, config, mode='train', run_dir=None):
        self.mode = mode
        self.t = self.steps = self.resets = 0
        self.closed = False
        self.id = uuid.uuid4().hex
        self.action_space = SimpleNamespace(shape=(12,), discrete=True)
        self.observation_space = SimpleNamespace(spaces={
            'image': SimpleNamespace(shape=(64, 64, 3)),
            'heatmap': SimpleNamespace(shape=(64, 64, 1)),
        })
    def obs(self, first=False):
        scalar = dict(is_first=first, is_last=self.t >= 3, is_terminal=False,
                      jump=False, is_zoomed=False, is_calculated=False,
                      reward_on_zoomed=0., intrinsic=.01 * self.t,
                      intrinsic_on_zoomed=0., score=.1 * self.t,
                      score_on_zoomed=0., jumping_steps=1000., accumulated_reward=1000.)
        return dict(image=np.full((64, 64, 3), self.t * 32, np.uint8),
                    heatmap=np.full((64, 64), 128, np.uint8),
                    zoomed_image=None, heatmap_on_zoomed=None, **scalar)
    def reset(self):
        self.t = 0
        self.resets += 1
        self.id = uuid.uuid4().hex
        return self.obs(True)
    def step(self, action):
        assert action['action'].shape == (12,)
        self.steps += 1
        self.t += 1
        done = self.t >= 3
        return self.obs(), float(done), done, dict(real_done=done, success=done, first_success_step=3)
    def close(self):
        self.closed = True


def test_actual_ls_runner_budget_checkpoint_and_resume(tmp_path, monkeypatch):
    config = build_config(stage='smoke', logdir=tmp_path / 'run', overrides=[
        'device=cpu', 'env_device=cpu', 'min_free_gb=0', 'keep_checkpoints=10',
        'clean.steps=16', 'clean.prefill=8', 'clean.checkpoint_every=10',
        'clean.eval_every=7', 'clean.batch_length=4', 'clean.train_ratio=1',
        'clean.log_every=4'])
    monkeypatch.setattr(runner.tools, 'Logger', FakeLogger)
    def manifest(c):
        payload = copy.deepcopy(c)
        payload['clean'].pop('steps')
        return dict(identity=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    monkeypatch.setattr(runner, 'experiment_manifest', manifest)
    envs = []
    def factory(*args, **kwargs):
        env = FakeEnv(*args, **kwargs)
        envs.append(env)
        return env
    trainer = runner.Trainer(config, env_factory=factory)
    trainer.run()
    assert trainer.step == trainer.agent._step == 16
    assert envs[0].steps == 16
    assert len(envs) == 2 and all(env.closed for env in envs)
    assert trainer.agent._update_count > 0
    assert (trainer.dir / 'checkpoints/step_0000000010/manifest.json').exists()
    saved = trainer.checkpoints.load(trainer.meta['identity'])
    assert saved['step'] == 16
    assert saved['optimizers']['model']['optimizer']['state']
    before = len(saved['train_eps'])
    with pytest.raises(FileExistsError):
        runner.Trainer(config, env_factory=factory).run()
    resumed_config = copy.deepcopy(config)
    resumed_config['clean']['steps'] = 24
    restored = runner.Trainer(resumed_config, resume=True, env_factory=factory)
    restored.run()
    assert restored.step == restored.agent._step == 24
    assert envs[2].steps == 8
    assert len(restored.train_eps) >= before
    events = [json.loads(line) for line in (trainer.dir / 'runtime.jsonl').read_text().splitlines()]
    assert any(row['event'] == 'resume_boundary' and row['step'] == 16 for row in events)
    assert (trainer.dir / 'checkpoints/step_0000000020/manifest.json').exists()


def test_ls_paths_are_stage_and_method_specific():
    formal = build_config()
    smoke = build_config(stage='smoke')
    assert formal['logdir'] != smoke['logdir']
    assert '/ls_imagine/formal/' in formal['logdir']
    assert formal['clean']['checkpoint_every'] == 20000
    assert formal['clean']['encoder']['cnn_keys'] == '^(image|heatmap)$'


@pytest.mark.parametrize('relative', ['runtime.jsonl', 'checkpoints/step_0000000020/manifest.json',
                                    'train_eps/old-1001.npz'])
def test_ls_refuses_existing_data_even_if_metadata_was_removed(tmp_path, relative):
    directory = tmp_path / 'existing'
    artifact = directory / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text('existing experiment data')
    config = build_config(stage='smoke', logdir=directory)
    with pytest.raises(FileExistsError, match='Run already exists'):
        runner.Trainer(config).run()
    assert artifact.read_text() == 'existing experiment data'
    assert not (directory / 'config.json').exists()
