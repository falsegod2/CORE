"""Persistent Clean environment and Evidence checkpoint integration checks."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clean_evidence import runner


class FakeEnv:
    num_actions = 12

    def __init__(self, config, mode='train', run_dir=None):
        self.mode = mode
        self.steps = self.resets = 0
        self.t = 0
        self.closed = False

    def packet(self, first=False):
        success = self.t == 3
        return dict(image=np.full((64, 64, 3), self.t * 17, np.uint8),
                    heatmap=np.full((64, 64), 128, np.uint8),
                    heatmap_valid=self.mode == 'train',
                    is_first=first, is_last=success, is_terminal=success,
                    done=success, success=success, task_reward=float(success),
                    clip_reward=0.01 * self.t, intrinsic=0.01 * self.t,
                    discount=float(not success), episode_step=self.t,
                    first_success_step=2 if success else 3)

    def reset(self):
        self.resets += 1
        self.t = 0
        return self.packet(True)

    def step(self, action):
        assert 0 <= action < self.num_actions
        self.steps += 1
        self.t += 1
        return self.packet()

    def close(self):
        self.closed = True


class FakeMethod:
    def __init__(self, config, logdir):
        self.records = []
        self.updates = 0
        self.evaluation_updates = []
        self.carries = []
        self.active = False

    def act(self, obs, carry, training=True, random_action=None):
        if training:
            self.carries.append(carry)
        else:
            self.evaluation_updates.append(self.updates)
        return int(1 if random_action is None else random_action), (carry or 0) + 1, {}

    def begin_episode(self, obs):
        assert obs['is_first']
        assert not self.active
        self.active = True

    def record(self, previous, action, current, extras):
        assert self.active
        self.records.append((int(action), int(current['episode_step'])))

    def finish_episode(self):
        self.active = False

    def ready(self):
        return bool(self.records)

    def update(self):
        self.updates += 1
        return {'model_loss': float(self.updates)}

    def flush_metrics(self):
        return {}

    def state_dict(self):
        return copy.deepcopy(dict(records=self.records, updates=self.updates))

    def load_state_dict(self, state):
        self.records = copy.deepcopy(state['records'])
        self.updates = state['updates']
        self.active = False

    def evaluation(self, seed):
        raise AssertionError('Clean environment evaluation must not use an isolated seeded RNG')

    def parameter_report(self):
        return {'identity': 'test-only method'}

    def close(self):
        pass


def config(tmp_path, **changes):
    result = dict(logdir=str(tmp_path / 'run'), task='minedojo_harvest_log_in_plains',
                  seed=9, device='cpu', env_device='cpu', num_actions=12,
                  train_steps=11, prefill=3, pretrain_updates=2,
                  batch_size=1, batch_length=2, train_ratio=1,
                  eval_every=4, eval_episodes=1, log_every=2,
                  checkpoint_every=4, keep_checkpoints=10, min_free_gb=0,
                  max_episode_steps=3, protocol={'eval_action': 'mode'})
    result.update(changes)
    return result


@pytest.fixture
def manifest(monkeypatch):
    def test_manifest(config):
        payload = {key: value for key, value in config.items() if key != 'train_steps'}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return {'identity': digest, 'test_only': True}
    monkeypatch.setattr(runner, 'experiment_manifest', test_manifest)


def rows(trainer):
    return [json.loads(line) for line in (trainer.dir / 'metrics.jsonl').read_text().splitlines()]


def factories():
    environments = []
    def make_env(*args, **kwargs):
        env = FakeEnv(*args, **kwargs)
        environments.append(env)
        return env
    return environments, make_env


def test_persistent_train_eval_initial_evaluation_precedes_pretrain_and_budget_is_exact(tmp_path, manifest):
    environments, make_env = factories()
    trainer = runner.Trainer(config(tmp_path), env_factory=make_env, method_factory=FakeMethod)
    trainer.run()
    assert trainer.step == 11 and trainer.updates == 6
    assert len(environments) == 2
    train, evaluation = environments
    assert train.mode == 'train' and evaluation.mode == 'eval'
    assert train.steps == 11 and evaluation.steps == 9
    assert train.closed and evaluation.closed
    assert trainer.method.evaluation_updates[:3] == [0, 0, 0]
    assert len(trainer.method.records) == 11
    events = rows(trainer)
    assert [row['step'] for row in events if row['event'] == 'evaluation'] == [3, 7, 11]
    assert [row['step'] for row in events if row['event'] == 'checkpoint'] == [4, 8, 11]
    # Evaluation at step7 happens inside a real training episode; its carry is
    # not cleared, while genuine reset rows start with carry=None.
    assert trainer.method.carries[6:9] == [None, 1, 2]
    train_logs = [row for row in events if row['event'] == 'train']
    assert sum(row.get('real_action_steps', 0) for row in train_logs) == 11
    assert sum(row.get('prefill_action_steps', 0) for row in train_logs) == 3
    for row in events:
        if row['event'] == 'evaluation':
            assert sum(row['eval_action_counts']) == row['eval_actual_steps']
            assert row['eval_action_mode'] == 1
            assert row['eval_action_mode_fraction'] == 1.


def test_resume_appends_metrics_and_continues_checkpoints_and_scheduling(tmp_path, manifest):
    environments, make_env = factories()
    original = config(tmp_path)
    first = runner.Trainer(original, env_factory=make_env, method_factory=FakeMethod)
    first.run()
    old_rows = rows(first)
    old_checkpoint = (first.dir / 'checkpoints/step_0000000011/state.pkl.gz').read_bytes()
    extended = dict(original, train_steps=17)
    resumed = runner.Trainer(extended, resume=True, env_factory=make_env, method_factory=FakeMethod)
    resumed.run()
    events = rows(resumed)
    new = events[len(old_rows):]
    assert events[:len(old_rows)] == old_rows
    assert resumed.step == 17 and resumed.updates == 9
    assert len(resumed.method.records) == 17
    assert len(environments) == 4
    assert [row['step'] for row in new if row['event'] == 'resume'] == [11]
    assert [row['step'] for row in new if row['event'] == 'checkpoint'] == [12, 16, 17]
    assert [row['step'] for row in new if row['event'] == 'evaluation'] == [15, 17]
    assert (first.dir / 'checkpoints/step_0000000011/state.pkl.gz').read_bytes() == old_checkpoint
    # Historical source constructor messages can start at zero; interactions
    # and saved global training counters must not rewind.
    assert all(row['step'] >= 11 for row in new if row['event'] not in ('learner_initializing', 'learner_ready'))
    assert sum(row.get('real_action_steps', 0) for row in new if row['event'] == 'train') == 6


def test_checkpoint_keeps_unlogged_action_population(tmp_path, manifest):
    cfg = config(tmp_path, train_steps=5, log_every=3)
    trainer = runner.Trainer(cfg, env_factory=FakeEnv, method_factory=FakeMethod)
    trainer.run()
    state = trainer.checkpoints.load(trainer.meta['identity'], step=4)
    assert state['action_counts'].sum() == 1
    assert state['action_counts'][1] == 1
    assert state['prefill_action_steps'] == 0


def test_existing_run_and_changed_identity_are_rejected_without_rewriting_logs(tmp_path, manifest):
    cfg = config(tmp_path)
    runner.Trainer(cfg, env_factory=FakeEnv, method_factory=FakeMethod).run()
    metric = Path(cfg['logdir']) / 'metrics.jsonl'
    before = metric.read_bytes()
    with pytest.raises(FileExistsError):
        runner.Trainer(cfg, env_factory=FakeEnv, method_factory=FakeMethod).run()
    assert metric.read_bytes() == before
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.Trainer(dict(cfg, seed=10), resume=True, env_factory=FakeEnv, method_factory=FakeMethod).run()
    assert metric.read_bytes() == before


@pytest.mark.parametrize('metrics_state', ['missing', 'empty'])
def test_missing_or_empty_metrics_does_not_allow_existing_state_to_be_overwritten(tmp_path, manifest, metrics_state):
    cfg = config(tmp_path)
    trainer = runner.Trainer(cfg, env_factory=FakeEnv, method_factory=FakeMethod)
    trainer.run()
    protected = [trainer.dir / 'config.json', trainer.dir / 'provenance.json',
                 trainer.dir / 'checkpoints/latest.json',
                 trainer.dir / 'checkpoints/step_0000000011/manifest.json',
                 trainer.dir / 'checkpoints/step_0000000011/state.pkl.gz']
    before = {path: path.read_bytes() for path in protected}
    metrics = trainer.dir / 'metrics.jsonl'
    if metrics_state == 'missing':
        metrics.unlink()
    else:
        metrics.write_text('')
    # A different seed makes an accidental provenance rewrite observable.
    with pytest.raises(FileExistsError, match='Run already exists'):
        runner.Trainer(dict(cfg, seed=10), env_factory=FakeEnv, method_factory=FakeMethod).run()
    assert all(path.read_bytes() == data for path, data in before.items())
    if metrics_state == 'missing':
        assert not metrics.exists()
    else:
        assert metrics.read_bytes() == b''


def test_final_threshold_checkpoint_contains_final_eval_rng_and_finished_replay(tmp_path, manifest):
    cfg = config(tmp_path, train_steps=12)
    trainer = runner.Trainer(cfg, env_factory=FakeEnv, method_factory=FakeMethod)
    trainer.run()
    saved = trainer.checkpoints.load(trainer.meta['identity'])
    assert saved['step'] == 12
    assert saved['last_eval'] == 12
    assert saved['evaluations'] == 4
    assert saved['next_checkpoint'] == 16


def test_real_evidence_small_model_resume_uses_clean_runner_and_preserves_replay(tmp_path, manifest):
    pytest.importorskip('torch')
    from clean_evidence.configuration import build_config
    cfg = build_config(stage='smoke', logdir=str(tmp_path / 'real-run'), overrides=[
        'device=cpu', 'env_device=cpu', 'train_steps=16', 'prefill=8',
        'batch_size=1', 'batch_length=4', 'pretrain_updates=1',
        'eval_every=8', 'eval_episodes=1', 'checkpoint_every=8',
        'log_every=4', 'min_free_gb=0', 'train_ratio=1'])
    # Fake physics supplies genuine RGB/heatmap/reward packets; the learner,
    # losses, optimizers, replay, checkpoint and Trainer are the real code.
    first = runner.Trainer(cfg, env_factory=FakeEnv)
    first.run()
    assert first.step == 16 and first.updates == 3
    old = rows(first)
    resumed = runner.Trainer(dict(cfg, train_steps=24), resume=True, env_factory=FakeEnv)
    resumed.run()
    new = rows(resumed)[len(old):]
    assert resumed.step == 24 and resumed.updates == 5
    assert resumed.method.replay.steps() == 24
    assert [row['step'] for row in new if row['event'] == 'resume'] == [16]
    assert [row['step'] for row in new if row['event'] == 'checkpoint'] == [24]
    assert [row['step'] for row in new if row['event'] == 'evaluation'] == [24]
