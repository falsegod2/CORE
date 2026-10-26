"""Read-only evidence broadcast preserves native imagined trajectories."""
import torch

from clean_evidence.vendor.dreamerv3_torch.models import ImagBehavior
from clean_evidence.methods.evidence_behavior import EvidenceImagBehavior
from evidence_tests.test_model_core import build


def test_broadcast_evidence_has_native_actions_features_and_targets(tmp_path):
    _, method, env = build(tmp_path)
    _, carry, _ = method.act(env.reset(), None)
    start = {key: value[None].detach() for key, value in carry[0].items()}
    assert isinstance(method.behavior, EvidenceImagBehavior)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        optimized = method.behavior._imagine(start, method.behavior.actor, 5)
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        reference = ImagBehavior._imagine(method.behavior, start, method.behavior.actor, 5)
    assert torch.equal(final_rng, torch.get_rng_state())
    assert torch.equal(optimized[0], reference[0])
    assert torch.equal(optimized[2], reference[2])
    for key in reference[1]:
        assert torch.equal(optimized[1][key], reference[1][key]), key
    for key in ('mem_tokens', 'mem_valid', 'mem_age'):
        view = optimized[1][key]
        assert view.stride(0) == 0
        assert view.untyped_storage().data_ptr() == start[key].untyped_storage().data_ptr()
    with torch.no_grad():
        fast_reward = method.wm.policy_reward(method.wm.dynamics.get_feat(optimized[1]))
        native_reward = method.wm.policy_reward(method.wm.dynamics.get_feat(reference[1]))
        fast = method.behavior._compute_target(optimized[0], optimized[1], fast_reward)
        native = method.behavior._compute_target(reference[0], reference[1], native_reward)
    for actual, expected in zip(fast[0], native[0]):
        assert torch.equal(actual, expected)
    assert torch.equal(fast[1], native[1])


def test_empty_imagination_retains_memory_shapes(tmp_path):
    _, method, _ = build(tmp_path)
    initial = method.wm.dynamics.initial(1)
    start = {key: value[:0][None] for key, value in initial.items()}
    features, states, actions = method.behavior._imagine(start, method.behavior.actor, 5)
    assert features.shape[:2] == actions.shape[:2] == (5, 0)
    assert states['mem_tokens'].shape == (5, 0, method.wm.dynamics.memory_capacity,
                                          method.wm.dynamics.memory_dim)
