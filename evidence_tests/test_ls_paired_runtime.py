"""Managed LS exposes comparison metadata without duplicate scalar evaluations."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(1)

from evidence_tests.test_ls_runner import FakeEnv, FakeLogger
from clean_ls import runner
from clean_ls.configuration import build_config


class WritingLogger(FakeLogger):
    def __init__(self, config, logdir, step):
        super().__init__(config, logdir, step)
        self.path = Path(logdir) / 'metrics.jsonl'

    def write(self, *args, **kwargs):
        row = dict(step=kwargs.get('step', self.step), **self.values)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.values.clear()


def _manifest(config):
    payload = copy.deepcopy(config)
    payload['clean'].pop('steps')
    return {'identity': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def test_ls_complete_metrics_and_resume_schedule(tmp_path, monkeypatch):
    config = build_config(stage='smoke', logdir=tmp_path / 'run', overrides=[
        'device=cpu', 'env_device=cpu', 'min_free_gb=0', 'keep_checkpoints=10',
        'clean.steps=16', 'clean.prefill=8', 'clean.checkpoint_every=10',
        'clean.eval_every=7', 'clean.batch_length=4', 'clean.train_ratio=1',
        'clean.log_every=4'])
    monkeypatch.setattr(runner.tools, 'Logger', WritingLogger)
    monkeypatch.setattr(runner, 'experiment_manifest', _manifest)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    environments = []
    def factory(*args, **kwargs):
        env = FakeEnv(*args, **kwargs)
        environments.append(env)
        return env
    trainer = runner.Trainer(config, env_factory=factory)
    trainer.run()
    records = [json.loads(line) for line in (trainer.dir / 'metrics.jsonl').read_text().splitlines()]
    scalar = [r for r in records if 'eval_success' in r]
    events = [r for r in records if r.get('event') == 'evaluation']
    assert [r['step'] for r in scalar] == [8, 15, 16]
    assert len(scalar) == len(events)
    for native, event in zip(scalar, events):
        assert 'eval_success' not in event
        assert event['result']['eval_success'] == native['eval_success'] == 1
        assert event['result']['eval_return'] == native['eval_return'] == 1
        assert event['eval_actual_steps'] == native['eval_actual_steps'] == 3
        assert native['eval_episodes'] == 1
        assert native['eval_seconds'] >= 0
        assert event['eval_seconds'] >= native['eval_seconds']
    launch = next(r for r in records if r.get('event') == 'launch')
    assert launch['runtime'] == {'cuda_visible_devices': '1', 'logical_device': 'cpu', 'device_name': None}
    assert all(r['checkpoint_seconds'] >= 0 for r in records if r.get('event') == 'checkpoint')
    assert environments[0].steps == 16
    assert environments[1].steps == 9

    extended = copy.deepcopy(config)
    extended['clean']['steps'] = 24
    runner.Trainer(extended, resume=True, env_factory=factory).run()
    resumed = [json.loads(line) for line in (trainer.dir / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in resumed if 'eval_success' in r] == [8, 15, 16, 22, 24]
    assert environments[2].steps == 8
    assert environments[3].steps == 6
    assert len([r for r in resumed if r.get('event') == 'resume']) == 1
    assert len([r for r in resumed if r.get('event') == 'completed']) == 2
    assert all(env.closed for env in environments)


def test_observed_logger_preserves_native_fields_and_step(tmp_path):
    native = WritingLogger(None, tmp_path, 3)
    logger = runner._ObservedLogger(native)
    logger.step = 17
    logger.scalar('train_success', 1)
    logger.write(step=17)
    assert native.step == 17
    row = json.loads((tmp_path / 'metrics.jsonl').read_text())
    assert row == {'step': 17, 'train_success': 1.0}
    assert logger.last_evaluation is None
    assert logger._writer is native._writer
