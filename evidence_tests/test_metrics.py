"""Diagnostics use population weights and do not alter learning trajectories."""
import copy

import numpy as np
import pytest

from clean_evidence.metrics import MeanMetrics, action_summary


def test_rare_positive_mae_uses_fractional_weights_and_ignores_empty_updates():
    metrics = MeanMetrics()
    name = 'rc_positive_abs_error'
    metrics.add({name + '_numerator': 0.25, name + '_denominator': 0.25})
    for _ in range(100):
        metrics.add({name + '_numerator': 0., name + '_denominator': 0.})
    metrics.add({name + '_numerator': 6., name + '_denominator': 2.})
    result = metrics.pop()
    assert result[name] == pytest.approx(6.25 / 2.25)
    assert result[name + '_numerator'] == 6.25
    assert result[name + '_denominator'] == 2.25
    assert metrics.pop() == {}


def test_no_positive_examples_is_null_and_regular_metrics_keep_mean():
    metrics = MeanMetrics()
    for loss in (2., 4.):
        metrics.add({'rc_positive_abs_error_numerator': 0.,
                     'rc_positive_abs_error_denominator': 0., 'model_loss': loss})
    result = metrics.pop()
    assert result['rc_positive_abs_error'] is None
    assert result['model_loss'] == 3.


def test_action_histogram_exposes_constant_downward_policy():
    counts = np.zeros(12, dtype=np.int64)
    counts[3] = 100
    result = action_summary(counts)
    assert result['real_action_steps'] == 100
    assert result['real_action_mode'] == 3
    assert result['real_action_mode_fraction'] == 1.
    assert sum(result['real_action_fractions']) == 1.


def test_reward_head_diagnostics_preserve_native_update_and_rng(tmp_path):
    torch = pytest.importorskip('torch')
    from clean_evidence.vendor.dreamerv3_torch.models import ImagBehavior
    from evidence_tests.test_model_core import build, assert_same_state

    _, method, env = build(tmp_path)
    _, carry, _ = method.act(env.reset(), None)
    start = {key: value[None].detach() for key, value in carry[0].items()}
    saved = copy.deepcopy(method.state_dict())
    objective = lambda feat, state, action: method.wm.policy_reward(
        method.wm.dynamics.get_feat(state))
    native = ImagBehavior._train(method.behavior, start, objective)
    expected_state = copy.deepcopy(method.behavior.state_dict())
    expected_rng = torch.get_rng_state().clone()
    method.load_state_dict(saved)
    result = method.behavior._train(start, objective)
    assert_same_state(expected_state, method.behavior.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())
    for key, expected in native[-1].items():
        np.testing.assert_array_equal(result[-1][key], expected)
    np.testing.assert_allclose(
        result[-1]['imag_reward_mean'],
        result[-1]['imag_task_reward_mean'] + result[-1]['imag_scaled_clip_reward_mean'],
        atol=1e-6)


def test_rc_fractional_positive_population_is_reported_without_clamping():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from clean_evidence.methods.evidence_model import EvidenceWorldModel

    options = dict(fc_weight=0., rc_weight=1., horizons=[1],
                   horizon_weights=[0.25], starts_per_sequence=1,
                   mc_samples=1, freeze_rc_reward_head=False)
    fake = SimpleNamespace(
        options=options, _config=SimpleNamespace(discount=0.99),
        dynamics=SimpleNamespace(
            img_step=lambda state, action: {'deter': action[:, :1]},
            get_feat=lambda state: state['deter']),
        heads={'task_reward': lambda features: SimpleNamespace(mode=lambda: features)})
    fake._valid_prefixes = EvidenceWorldModel._valid_prefixes
    fake._select_starts = lambda data: torch.tensor([[0, 0]])
    data = dict(action=torch.tensor([[[0.], [2.]]]),
                task_reward=torch.tensor([[[0.], [1.]]]),
                is_first=torch.tensor([[True, False]]),
                is_terminal=torch.tensor([[False, True]]),
                is_last=torch.tensor([[False, True]]),
                sequence_valid=torch.ones(1, 2), loss_mask=torch.ones(1, 2))
    losses, values = EvidenceWorldModel._auxiliary_losses(fake, {'deter': torch.zeros(1, 2, 1)}, data)
    # Preserve the existing RC objective, while fixing only its MAE diagnostic.
    assert losses['rc'].item() == pytest.approx(0.25)
    metrics = MeanMetrics()
    metrics.add(values)
    result = metrics.pop()
    assert result['rc_positive_abs_error'] == pytest.approx(1.)
    assert result['rc_positive_abs_error_denominator'] == pytest.approx(0.25)
    assert result['rc_zero_abs_error'] is None
