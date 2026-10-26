"""Memory probes inspect the actual read without changing policy computation."""
import math

import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(1)

from evidence_tests.test_model_core import make_rssm, assert_same_state


def populated(model, batch=2):
    state = {key: value.detach() for key, value in model.initial(batch).items()}
    state['mem_tokens'] = torch.arange(
        batch * model.memory_capacity * model.memory_dim,
        dtype=torch.float32).reshape(batch, model.memory_capacity, model.memory_dim) / 100
    state['mem_valid'][:, :4] = True
    state['mem_age'] = torch.arange(model.memory_capacity, dtype=torch.float32)[None].expand(batch, -1).clone()
    return state


def ratio(stats, name, prefix='probe'):
    den = stats[prefix + '_' + name + '_denominator']
    return stats[prefix + '_' + name + '_numerator'] / den if den else None


def test_capture_preserves_sampled_outputs_rng_gradients_and_parameters():
    model = make_rssm()
    state = populated(model)
    original_state = {k: v.detach().clone() for k, v in state.items()}
    parameters = tuple(model.parameters())
    original_parameters = [p.detach().clone() for p in parameters]
    rng = torch.get_rng_state().clone()
    action = torch.tensor([[1., 0., 0.]] * 2)

    def forward():
        result = state
        for _ in range(3):
            result = model.img_step(result, action, sample=True)
        loss = result['logit'].square().mean() + result['deter'].square().mean()
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        return result, grads, torch.get_rng_state().clone()

    plain, plain_grads, plain_rng = forward()
    torch.set_rng_state(rng)
    with model.capture_memory_diagnostics(prefix='probe') as stats:
        captured, captured_grads, captured_rng = forward()
    assert_same_state(plain, captured)
    assert_same_state(state, original_state)
    assert torch.equal(plain_rng, captured_rng)
    for a, b in zip(plain_grads, captured_grads):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b)
    for actual, saved in zip(parameters, original_parameters):
        assert torch.equal(actual, saved)
        assert actual.grad is None
    assert all(type(x) in (float, int) and math.isfinite(x) for x in stats.values())
    assert stats['probe_transition_rows'] == 6
    assert stats['probe_active_read_rows'] == 6
    assert ratio(stats, 'active_read_fraction') == 1
    assert model._memory_diagnostics is None


def test_attention_null_valid_mass_entropy_and_age_match_uniform_reference():
    model = make_rssm()
    state = populated(model)
    state['mem_valid'][1] = True
    with torch.no_grad():
        size = model.memory_dim
        model.attention.in_proj_weight[:2 * size].zero_()
        model.attention.in_proj_bias[:2 * size].zero_()
        model.read_gate.weight.zero_()
        model.read_gate.bias.fill_(math.log(.25 / .75))
    with model.capture_memory_diagnostics(prefix='probe') as stats:
        model.img_step(state, torch.tensor([[1., 0., 0.]] * 2), sample=False)
    expected_null = (1 / 5 + 1 / 9) / 2
    expected_real = (4 / 5 + 8 / 9) / 2
    expected_age = (sum(range(1, 5)) / 5 + sum(range(1, 9)) / 9) / (4 / 5 + 8 / 9)
    assert ratio(stats, 'attention_null_mass') == pytest.approx(expected_null)
    assert ratio(stats, 'attention_real_mass') == pytest.approx(expected_real)
    assert ratio(stats, 'attention_entropy_normalized') == pytest.approx(1.0)
    assert ratio(stats, 'attended_age_steps') == pytest.approx(expected_age)
    assert ratio(stats, 'gate_mean') == pytest.approx(.25)
    assert ratio(stats, 'gate_low_fraction') == 0
    assert ratio(stats, 'gate_high_fraction') == 0


@pytest.mark.parametrize('enabled,mode', [(False, 'none'), (True, 'none'), (True, 'each_step')])
def test_empty_or_disabled_memory_does_not_claim_measured_gate_or_attention(enabled, mode):
    model = make_rssm(memory_enabled=enabled, read_mode=mode)
    state = model.initial(2)
    with model.capture_memory_diagnostics(prefix='probe') as stats:
        model.img_step(state, torch.tensor([[1., 0., 0.]] * 2), sample=False)
    assert stats['probe_enabled'] == int(enabled)
    assert stats['probe_transition_rows'] == 2
    assert stats['probe_active_read_rows'] == 0
    assert ratio(stats, 'gate_mean') is None
    assert ratio(stats, 'attention_real_mass') is None
    assert ratio(stats, 'available_fraction') == 0
    assert ratio(stats, 'active_read_fraction') == 0


def test_start_only_records_only_rows_that_read_and_counts_skipped_transitions():
    model = make_rssm(read_mode='start_only')
    state = populated(model)
    state['imag_steps'][1] = 2
    with model.capture_memory_diagnostics(prefix='probe') as stats:
        state = model.img_step(state, torch.tensor([[1., 0., 0.]] * 2), sample=False)
        model.img_step(state, torch.tensor([[1., 0., 0.]] * 2), sample=False)
    assert stats['probe_read_mode_code'] == 1
    assert stats['probe_transition_rows'] == 4
    assert stats['probe_active_read_rows'] == 1
    assert ratio(stats, 'active_read_fraction') == .25
    assert stats['probe_gate_mean_denominator'] == model._deter // 2
    assert stats['probe_attention_null_mass_denominator'] == model.attention.num_heads


def test_nested_and_failed_capture_restore_collector():
    model = make_rssm()
    with pytest.raises(RuntimeError, match='Nested'):
        with model.capture_memory_diagnostics(prefix='outer'):
            with model.capture_memory_diagnostics(prefix='inner'):
                pass
    assert model._memory_diagnostics is None


def test_row_mask_excludes_invalid_prefixes_without_changing_rollout_or_gradients():
    model = make_rssm()
    state = populated(model)
    # Distinct ages make accidental inclusion of the second row observable.
    state['mem_age'][1] += 100
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    action = torch.tensor([[1., 0., 0.]] * 2)

    def forward():
        result = model.img_step(state, action, sample=True)
        grads = torch.autograd.grad(result['logit'].square().mean(), parameters, allow_unused=True)
        return result, grads, torch.get_rng_state().clone()

    with model.capture_memory_diagnostics(prefix='all') as all_stats:
        plain, plain_grads, plain_rng = forward()
    torch.set_rng_state(rng)
    with model.capture_memory_diagnostics(prefix='probe') as stats:
        with model.memory_diagnostic_mask(torch.tensor([True, False])):
            masked, masked_grads, masked_rng = forward()
    assert_same_state(plain, masked)
    assert torch.equal(plain_rng, masked_rng)
    for before, after in zip(plain_grads, masked_grads):
        assert (before is None) == (after is None)
        if before is not None:
            assert torch.equal(before, after)
    assert all_stats['all_transition_rows'] == 2
    assert stats['probe_transition_rows'] == 1
    assert stats['probe_active_read_rows'] == 1
    assert stats['probe_gate_mean_denominator'] == model._deter // 2
    assert stats['probe_attention_null_mass_denominator'] == model.attention.num_heads
    assert 1 <= ratio(stats, 'attended_age_steps') <= 4
    assert model._memory_diagnostic_rows is None
    assert all(parameter.grad is None for parameter in parameters)


def test_row_mask_nested_scope_all_invalid_and_exception_cleanup():
    model = make_rssm()
    state = populated(model)
    action = torch.tensor([[1., 0., 0.]] * 2)
    outer = torch.tensor([True, False])
    with model.capture_memory_diagnostics(prefix='probe') as stats:
        with model.memory_diagnostic_mask(outer):
            model.img_step(state, action, sample=False)
            with pytest.raises(RuntimeError, match='intentional'):
                with model.memory_diagnostic_mask(torch.tensor([False, False])):
                    model.img_step(state, action, sample=False)
                    raise RuntimeError('intentional test failure')
            assert torch.equal(model._memory_diagnostic_rows, outer)
            model.img_step(state, action, sample=False)
    assert stats['probe_transition_rows'] == 2
    assert stats['probe_active_read_rows'] == 2
    assert model._memory_diagnostic_rows is None
    with pytest.raises(ValueError, match='boolean'):
        with model.memory_diagnostic_mask(torch.tensor([1, 0])):
            pass
    with pytest.raises(ValueError, match='match'):
        with model.capture_memory_diagnostics():
            with model.memory_diagnostic_mask(torch.tensor([True])):
                model.img_step(state, action, sample=False)
    assert model._memory_diagnostic_rows is None
    assert model._memory_diagnostics is None
    with model.capture_memory_diagnostics(prefix='again') as stats:
        assert stats['again_transition_rows'] == 0
    assert model._memory_diagnostics is None
