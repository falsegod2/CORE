"""Fixed-bank persistence, noninterference, and failure isolation."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from clean_evidence.diagnostics import FixedReplayProbes
from clean_evidence.methods.evidence_dreamer import EvidenceDreamer
from clean_evidence.methods.evidence_replay import EvidenceReplay
from clean_evidence.configuration import build_config, experiment_manifest
from clean_evidence.runner import Trainer
from evidence_tests.test_model_core import build, small_config, SyntheticEnv, assert_same_state
from evidence_tests.test_runner import config as runner_config, FakeEnv, FakeMethod, manifest


def diagnostic_config():
    config = small_config()
    config['diagnostics'] = dict(enabled=True, every=4, batch_size=2, gradients=True)
    return config


def assert_rng_equal(a, b):
    assert a[0] == b[0] and a[2:] == b[2:]
    np.testing.assert_array_equal(a[1], b[1])


def test_diagnostic_probes_leave_next_training_update_bit_exact(tmp_path):
    config, method, _ = build(tmp_path, config=diagnostic_config())
    method.update()
    saved = method.state_dict()
    expected = method.update()
    weights = copy.deepcopy(method.wm.state_dict())
    actor = copy.deepcopy(method.behavior.state_dict())
    method.load_state_dict(saved)
    # Existing .grad buffers must also survive the separate autograd assay.
    grads = [None if p.grad is None else p.grad.clone() for p in method.wm.parameters()]
    torch_rng = torch.get_rng_state().clone()
    np_rng, py_rng = np.random.get_state(), random.getstate()
    replay_rng = copy.deepcopy(method.replay.rng.get_state())
    report = method.diagnostics(20)
    assert not report['pending_banks']
    assert report['updates_performed'] == 0 and not report['weights_changed']
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert_rng_equal(np_rng, np.random.get_state())
    assert_rng_equal(replay_rng, method.replay.rng.get_state())
    assert py_rng == random.getstate()
    for p, before in zip(method.wm.parameters(), grads):
        assert (p.grad is None) if before is None else torch.equal(p.grad, before)
    actual = method.update()
    assert_same_state(weights, method.wm.state_dict())
    assert_same_state(actor, method.behavior.state_dict())
    for key in expected:
        np.testing.assert_allclose(expected[key], actual[key], atol=0, rtol=0)


def test_probes_are_fixed_checkpointed_and_positive_bank_is_nonempty(tmp_path):
    config, method, env = build(tmp_path, config=diagnostic_config())
    config['diagnostics']['gradients'] = False
    first = method.diagnostics(20)
    positive = first['banks']['reward_nonzero']['metrics']
    assert positive['rc_all_horizons_nonzero_prefixes'] > 0
    assert positive['rc_all_horizons_nonzero_target_mean'] > 0
    saved = method.state_dict()
    clone = EvidenceDreamer(config, tmp_path / 'clone')
    clone.load_state_dict(saved)
    # Alter training replay after capture; immutable diagnostic arrays cannot
    # change when replay is evicted, extended, or restored.
    clone.replay.episodes.clear()
    again = clone.diagnostics(40)
    for name in first['banks']:
        assert first['banks'][name]['fingerprint'] == again['banks'][name]['fingerprint']
        assert first['banks'][name]['captured_at_step'] == again['banks'][name]['captured_at_step'] == 20
        assert first['banks'][name]['metrics'] == again['banks'][name]['metrics']


def test_missing_positive_data_is_pending_not_fake_zero_error(tmp_path):
    config = diagnostic_config()
    config['diagnostics']['gradients'] = False
    method = EvidenceDreamer(config, tmp_path)
    env = SyntheticEnv(config)
    method.begin_episode(env.reset())
    for _ in range(8):
        method.record(None, 0, env.step(0), {})
    method.finish_episode()
    report = method.diagnostics(20)
    assert report['pending_banks'] == ['reward_nonzero']
    natural = report['banks']['natural']['metrics']
    assert natural['rc_all_horizons_nonzero_mae'] is None
    assert natural['rc_all_horizons_nonzero_rmse'] is None
    method.begin_episode(env.reset())
    for _ in range(6):
        method.record(None, 1, env.step(1), {})
    method.finish_episode()
    second = method.diagnostics(40)
    assert not second['pending_banks']
    assert second['banks']['natural']['fingerprint'] == report['banks']['natural']['fingerprint']
    assert second['banks']['reward_nonzero']['captured_at_step'] == 40


def test_exception_restores_modes_rng_and_parameter_flags(tmp_path, monkeypatch):
    _, method, _ = build(tmp_path, config=diagnostic_config())
    old_modes = [m.training for m in method.wm.modules()]
    torch_rng = torch.get_rng_state().clone()
    def fail(*a, **kw):
        torch.rand(8)
        raise RuntimeError('test diagnostic failure')
    monkeypatch.setattr(method.wm, 'diagnostic_metrics', fail)
    with pytest.raises(RuntimeError, match='test diagnostic failure'):
        method.diagnostics(20)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert old_modes == [m.training for m in method.wm.modules()]


def test_runner_schedules_and_restores_diagnostics_without_repeating(tmp_path, manifest):
    class DiagnosticFake(FakeMethod):
        def diagnostics(self, step):
            return dict(banks={}, updates_performed=0)
    config = runner_config(tmp_path, train_steps=8, diagnostics=dict(enabled=True, every=4, batch_size=2, gradients=False))
    Trainer(config, env_factory=FakeEnv, method_factory=DiagnosticFake).run()
    config['train_steps'] = 12
    Trainer(config, resume=True, env_factory=FakeEnv, method_factory=DiagnosticFake).run()
    rows = [json.loads(x) for x in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows if r['event'] == 'diagnostics'] == [4, 8, 12]
    checkpoints = [r for r in rows if r['event'] == 'checkpoint']
    assert all(r['checkpoint_seconds'] >= 0 for r in checkpoints)


def test_optional_diagnostic_failure_does_not_abort_training(tmp_path, manifest):
    class DiagnosticFake(FakeMethod):
        def diagnostics(self, step):
            raise RuntimeError('broken optional probe')
    config = runner_config(tmp_path, train_steps=8, diagnostics=dict(enabled=True, every=4, batch_size=2, gradients=False))
    Trainer(config, env_factory=FakeEnv, method_factory=DiagnosticFake).run()
    rows = [json.loads(x) for x in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows if r['event'] == 'diagnostics_failed'] == [4, 8]
    assert not [r for r in rows if r['event'] == 'failed']
    assert rows[-1]['event'] == 'completed'


@pytest.mark.parametrize('expression', ['diagnostics.every=0', 'diagnostics.batch_size=17',
                                        'diagnostics.enabled=1', 'diagnostics.gradients=0'])
def test_invalid_diagnostic_options_rejected(expression):
    with pytest.raises(ValueError):
        build_config(overrides=[expression])


def test_readonly_diagnostic_controls_do_not_change_resume_identity(tmp_path):
    assets = tmp_path / 'assets'
    clip = assets / 'weights/mineclip_attn.pth'
    teacher = assets / 'affordance_map/finetune_unet/finetune_checkpoints/harvest_log_in_plains/swin_unet_checkpoint.pth'
    clip.parent.mkdir(parents=True)
    teacher.parent.mkdir(parents=True)
    clip.write_bytes(b'clip fixture')
    teacher.write_bytes(b'teacher fixture')
    first = build_config(stage='smoke', assets_root=assets)
    second = build_config(stage='smoke', assets_root=assets,
                          overrides=['diagnostics.enabled=false', 'diagnostics.gradients=false',
                                     'diagnostics.every=5000'])
    assert experiment_manifest(first)['identity'] == experiment_manifest(second)['identity']
    different_bank = build_config(stage='smoke', assets_root=assets,
                                  overrides=['diagnostics.batch_size=3'])
    assert experiment_manifest(first)['identity'] != experiment_manifest(different_bank)['identity']


def test_resume_changed_diagnostic_interval_uses_next_new_boundary(tmp_path, monkeypatch):
    from clean_evidence import runner
    monkeypatch.setattr(runner, 'experiment_manifest', lambda config: {'identity': 'fixed-test-identity'})
    class DiagnosticFake(FakeMethod):
        def diagnostics(self, step):
            return dict(banks={}, updates_performed=0)
    config = runner_config(tmp_path, train_steps=8,
                           diagnostics=dict(enabled=True, every=4, batch_size=2, gradients=True))
    Trainer(config, env_factory=FakeEnv, method_factory=DiagnosticFake).run()
    config['train_steps'] = 13
    config['diagnostics'].update(every=5, gradients=False)
    Trainer(config, resume=True, env_factory=FakeEnv, method_factory=DiagnosticFake).run()
    rows = [json.loads(x) for x in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows if r['event'] == 'diagnostics'] == [4, 8, 10, 13]


def test_positive_bank_aligns_short_episode_reward_with_scored_horizon(tmp_path):
    config = diagnostic_config()
    config['diagnostics']['gradients'] = False
    config['batch_length'] = 16
    config['evidence'].update(horizons=[1, 2, 4, 8, 15],
                              horizon_weights=[1., 1., .75, .5, .25], starts_per_sequence=1)
    _, method, _ = build(tmp_path, config=config)
    report = method.diagnostics(20)
    assert report['banks']['reward_nonzero']['metrics']['rc_all_horizons_nonzero_prefixes'] > 0


def test_memory_recorded_counts_only_rows_in_valid_scored_prefixes(tmp_path):
    from clean_evidence.methods.diagnostic_model import _fixed_starts
    config, method, _ = build(tmp_path, config=diagnostic_config())
    config['diagnostics']['gradients'] = False
    report = method.diagnostics(20)
    for name, bank in method.diagnostic_probes.banks.items():
        data = method.wm.preprocess(bank['batch'])
        starts = _fixed_starts(method.wm, data, config['evidence']['starts_per_sequence'])
        prefix = method.wm._valid_prefixes(data, starts, config['evidence']['horizons'])
        expected = 0
        for offset in range(1, max(config['evidence']['horizons']) + 1):
            active = torch.zeros(starts.shape[0], dtype=torch.bool)
            for index, horizon in enumerate(config['evidence']['horizons']):
                if horizon >= offset and config['evidence']['horizon_weights'][index] > 0:
                    active |= prefix[:, index]
            expected += int(active.sum())
        metrics = report['banks'][name]['metrics']
        assert metrics['memory_recorded_transition_rows'] == expected * config['evidence']['mc_samples']
