"""Probe invariants: no optimization, honest ablated losses and episode masks."""
import copy
import random

import numpy as np
import pytest
import torch

from clean_evidence.metrics import MeanMetrics
from clean_evidence.rng import isolated_rng
from clean_evidence.methods.diagnostic_model import _fixed_starts, _recorded_probe
from evidence_tests.test_model_core import build, small_config


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _probe(method, batch, gradients=True):
    with isolated_rng(7341):
        return method.wm.diagnostic_metrics(batch, gradients=gradients)


def test_probe_preserves_parameters_existing_gradients_optimizer_flags_rng_and_inputs(tmp_path):
    _, method, _ = build(tmp_path)
    method.update()  # Populate Adam moments before checking read-only behavior.
    batch = method.replay.sample()
    before_batch = copy.deepcopy(batch)
    params = list(method.wm.parameters())
    params[0].requires_grad_(True)  # Restore mixed flags, not merely all-false.
    params[0].grad = torch.full_like(params[0], 0.125)
    flags = [p.requires_grad for p in params]
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    weights = copy.deepcopy(method.wm.state_dict())
    optimizer = copy.deepcopy(method.wm._model_opt._opt.state_dict())
    options = copy.deepcopy(method.wm.options)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    first = _probe(method, batch)
    second = _probe(method, batch)
    assert first == second
    assert first['grad_base_norm'] > 0 and first['grad_fc_raw_norm'] > 0
    assert first['grad_fc_weighted_norm'] == pytest.approx(first['grad_fc_raw_norm'] * options['fc_weight'])
    assert first['grad_rc_weighted_norm'] == pytest.approx(first['grad_rc_raw_norm'] * options['rc_weight'])
    _equal(weights, method.wm.state_dict())
    _equal(optimizer, method.wm._model_opt._opt.state_dict())
    _equal(options, method.wm.options)
    _equal(before_batch, batch)
    _equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    assert flags == [p.requires_grad for p in params]
    for grad, parameter in zip(grads, params):
        if grad is None:
            assert parameter.grad is None
        else:
            assert torch.equal(grad, parameter.grad)


def test_disabled_objectives_still_have_calibration_metrics_but_zero_weighted_gradients(tmp_path):
    config = small_config(('evidence_no_fc', 'evidence_no_rc', 'evidence_no_memory'))
    _, method, _ = build(tmp_path, config)
    batch = method.replay.sample()
    # Ensure a nonzero actual target that a zero-initialized task head misses.
    batch['task_reward'][batch['sequence_valid']] = 1.0
    metrics = _probe(method, batch)
    assert metrics['fc_loss'] > 0
    assert metrics['rc_loss'] > 0
    assert metrics['grad_fc_weighted_norm'] == metrics['grad_rc_weighted_norm'] == 0
    assert metrics['rc_all_horizons_nonzero_mae_denominator'] > 0
    assert metrics['rc_all_horizons_nonzero_mae_numerator'] > 0
    assert metrics['rc_all_horizons_zero_mae_denominator'] == 0
    means = MeanMetrics()
    means.add(metrics)
    merged = means.pop()
    assert merged['rc_all_horizons_zero_mae'] is None
    assert merged['rc_all_horizons_nonzero_mae'] > 0


def test_probe_without_gradients_matches_values_and_supports_no_semantic(tmp_path):
    config = small_config()
    config['needs_heatmap'] = False
    config['evidence'].update(local_heat_weight=0.0, posterior_heat_weight=0.0, fc_heat_weight=0.0)
    _, method, _ = build(tmp_path, config)
    batch = method.replay.sample()
    batch.pop('heatmap')
    batch.pop('heatmap_valid')
    no_grad = _probe(method, batch, gradients=False)
    grad = _probe(method, batch, gradients=True)
    assert no_grad['fc_heat_available'] == 0
    assert no_grad['fc_heat_denominator'] == 0
    for key in ('base_loss', 'fc_loss', 'rc_loss', 'rc_all_horizons_all_target_mean_numerator'):
        assert no_grad[key] == pytest.approx(grad[key], rel=1e-6, abs=1e-7)
    assert 'grad_base_norm' not in no_grad


def test_fixed_starts_and_returns_do_not_cross_terminal_reset_or_padding(tmp_path):
    _, method, _ = build(tmp_path)
    model = method.wm
    # Only row 0 can start: terminal row 1 is a valid endpoint; later rows must
    # never contribute their deliberately huge reward to a prefix from row 0.
    batch = method.replay.sample()
    batch = {k: v[:1].copy() for k, v in batch.items()}
    batch['sequence_valid'][:] = False
    batch['sequence_valid'][0, :3] = True
    batch['loss_mask'][:] = False
    batch['loss_mask'][0, 0] = True
    batch['is_first'][:] = False
    batch['is_first'][0, 0] = True
    batch['is_first'][0, 2] = True
    batch['is_terminal'][:] = False
    batch['is_terminal'][0, 1] = True
    batch['is_last'][:] = False
    batch['is_last'][0, 1] = True
    batch['task_reward'][:] = 1000
    batch['task_reward'][0, 1] = 1
    metrics = _probe(method, batch, gradients=False)
    assert metrics['starts'] == 1
    assert metrics['rc_h1_all_prefixes'] == 1
    assert metrics['rc_h1_all_target_mean_numerator'] == 1
    assert metrics['rc_h1_all_target_mean_denominator'] == 1
    for horizon in (2, 4):
        assert metrics[f'rc_h{horizon}_all_prefixes'] == 0
        assert metrics[f'rc_h{horizon}_all_mae_denominator'] == 0


def test_requires_grad_flags_restore_if_probe_fails(tmp_path, monkeypatch):
    _, method, _ = build(tmp_path)
    batch = method.replay.sample()
    params = list(method.wm.parameters())
    params[0].requires_grad_(True)
    flags = [p.requires_grad for p in params]
    def fail(*args, **kwargs):
        raise RuntimeError('intentional probe exception')
    monkeypatch.setattr(method.wm.encoder, 'forward', fail)
    with pytest.raises(RuntimeError, match='intentional probe exception'):
        _probe(method, batch)
    assert flags == [p.requires_grad for p in params]


def test_fixed_probe_matches_training_auxiliary_estimator_for_identical_starts(tmp_path, monkeypatch):
    _, method, _ = build(tmp_path)
    model = method.wm
    data = model.preprocess(method.replay.sample())
    with torch.no_grad():
        embed = model.encoder(data)
        tokens, _ = model.writer(data['image'])
        post, _ = model.dynamics.observe(embed, data['action'], data['is_first'],
            memory_tokens=tokens, sequence_valid=data['sequence_valid'], loss_mask=data['loss_mask'])
        starts = _fixed_starts(model, data, model.options['starts_per_sequence'])
        monkeypatch.setattr(model, '_select_starts', lambda unused: starts)
        with isolated_rng(615):
            reference, _ = model._auxiliary_losses(post, data)
        with isolated_rng(615):
            measured, _ = _recorded_probe(model, post, data, starts)
    assert torch.equal(measured['fc'], reference['fc'])
    assert torch.equal(measured['rc'], reference['rc'])


def test_readonly_probe_does_not_change_next_training_update(tmp_path):
    _, left, _ = build(tmp_path / 'left')
    _, right, _ = build(tmp_path / 'right')
    # Constructors and replay sampling have independent deterministic seeds.
    fixed = left.replay.sample()
    right.replay.sample()
    with isolated_rng(999):
        reference = left.update()
    _probe(right, fixed, gradients=True)
    with isolated_rng(999):
        measured = right.update()
    assert reference == measured
    _equal(left.wm.state_dict(), right.wm.state_dict())
    _equal(left.behavior.state_dict(), right.behavior.state_dict())
    _equal(left.wm._model_opt._opt.state_dict(), right.wm._model_opt._opt.state_dict())
