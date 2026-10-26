"""CPU invariants for the candidate method; no Minecraft performance claims."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from clean_evidence.methods.evidence_dreamer import EvidenceDreamer
from clean_evidence.methods.evidence_replay import EvidenceReplay
from clean_evidence.methods.evidence_rssm import EvidenceRSSM
from clean_evidence.methods.evidence_model import EvidenceWorldModel

from clean_evidence.methods.evidence_model import DEFAULT_OPTIONS


def small_config(ablations=()):
    evidence = copy.deepcopy(DEFAULT_OPTIONS)
    evidence.update(memory_observations=2, memory_dim=16, attention_heads=2,
                    writer_depth=4, burnin=4, horizons=[1, 2, 4],
                    horizon_weights=[1., 1., .5], starts_per_sequence=2, mc_samples=2)
    for name in ablations:
        field, value = {'evidence_no_memory': ('memory_enabled', False),
                        'evidence_start_only': ('read_mode', 'start_only'),
                        'evidence_no_fc': ('fc_weight', 0.),
                        'evidence_no_rc': ('rc_weight', 0.)}[name]
        evidence[field] = value
    return dict(seed=31, device='cpu', env_device='cpu', num_actions=12,
                batch_size=2, batch_length=5, replay_capacity=100,
                protocol={'eval_action': 'mode'}, evidence=evidence,
                dreamer={'dynamics': 'evidence', 'both_action': True, 'loss_plugins': [],
                         'overrides': {'dyn_deter': 32, 'dyn_hidden': 16, 'dyn_stoch': 4,
                                       'dyn_discrete': 4, 'units': 32, 'imag_horizon': 5,
                                       'encoder': {'cnn_depth': 4, 'mlp_layers': 1, 'mlp_units': 32},
                                       'decoder': {'cnn_depth': 4, 'mlp_layers': 1, 'mlp_units': 32},
                                       'actor': {'layers': 1}, 'critic': {'layers': 1},
                                       'reward_head': {'layers': 1}, 'cont_head': {'layers': 1}}})


class SyntheticEnv:
    """Deterministic model-test packets, not a MineDojo implementation/fallback."""
    def __init__(self, config, needs_heatmap=True):
        self.heat = needs_heatmap
        self.t = 0
        self.action = 0

    def _obs(self, first=False):
        success = self.t == 6 and self.action == 1
        last = success or self.t >= 8
        return dict(image=np.full((64, 64, 3), self.t * 17 % 255, np.uint8),
                    heatmap=np.full((64, 64), 128, np.uint8),
                    heatmap_valid=self.heat, task_reward=float(success),
                    clip_reward=float(self.t) / 1000., intrinsic=float(self.t) / 1000.,
                    is_first=first, is_last=last, is_terminal=success, done=last,
                    discount=float(not success))

    def reset(self):
        self.t, self.action = 0, 0
        return self._obs(True)

    def step(self, action):
        self.t += 1
        self.action = action
        return self._obs()





def make_rssm(**overrides):
    config = dict(stoch=4, deter=16, hidden=16, discrete=3, embed=8,
                  num_actions=3, initial="learned", device="cpu", seed=21,
                  memory_observations=2, memory_dim=8, attention_heads=2)
    config.update(overrides)
    return EvidenceRSSM(**config)


def build(tmp_path, config=None):
    config = small_config() if config is None else config
    method = EvidenceDreamer(config, tmp_path)
    env = SyntheticEnv(config, needs_heatmap=True)
    for _ in range(3):
        previous = env.reset()
        method.begin_episode(previous)
        for _ in range(8):
            current = env.step(1)
            method.record(previous, 1, current, {})
            previous = current
            if current["done"]:
                break
        method.finish_episode()
    return config, method, env


def assert_finite(metrics):
    assert metrics
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        assert np.isfinite(value).all(), key


def assert_same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key




def test_current_rgb_evidence_does_not_leak_into_its_prior():
    model = make_rssm()
    embed = torch.arange(8, dtype=torch.float32)[None] / 10
    action = torch.zeros(1, 3)
    first = torch.ones(1)
    tokens_a = torch.ones(1, 4, 8)
    tokens_b = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 10
    post_a, prior_a = model.obs_step(None, action, embed, first,
                                    sample=False, memory_tokens=tokens_a)
    post_b, prior_b = model.obs_step(None, action, embed, first,
                                    sample=False, memory_tokens=tokens_b)
    assert_same_state(prior_a, prior_b)
    for key in ("deter", "stoch", "logit"):
        assert torch.equal(post_a[key], post_b[key])
    assert not torch.equal(post_a["mem_tokens"], post_b["mem_tokens"])
    assert not prior_a["mem_valid"].any()
    action = torch.tensor([[1., 0., 0.]])
    next_a = model.img_step(post_a, action, sample=False)
    next_b = model.img_step(post_b, action, sample=False)
    assert not torch.allclose(next_a["deter"][:, :8], next_b["deter"][:, :8])


def test_imagination_is_readonly_and_both_branches_receive_real_actions():
    model = make_rssm()
    state, _ = model.obs_step(None, None, torch.zeros(2, 8), torch.ones(2),
                              sample=False, memory_tokens=torch.randn(2, 4, 8))
    saved = {key: value.clone() for key, value in state.items()}
    left = model.img_step(state, torch.tensor([[1., 0., 0.]] * 2), sample=False)
    right = model.img_step(state, torch.tensor([[0., 1., 0.]] * 2), sample=False)
    assert not torch.allclose(left["deter"][:, :8], right["deter"][:, :8])
    assert not torch.allclose(left["deter"][:, 8:], right["deter"][:, 8:])
    rolled = left
    for _ in range(4):
        rolled = model.img_step(rolled, torch.tensor([[1., 0., 0.]] * 2), sample=False)
    for key in ("mem_tokens", "mem_valid", "mem_age"):
        assert torch.equal(rolled[key], saved[key]), key
    assert_same_state(state, saved)
    assert torch.equal(rolled["imag_steps"], torch.full((2, 1), 5.))


def test_real_reset_discards_prior_episode_evidence_without_mutation():
    model = make_rssm()
    tokens = torch.randn(2, 4, 8)
    state, _ = model.obs_step(None, None, torch.zeros(2, 8), torch.ones(2),
                              sample=False, memory_tokens=tokens)
    saved = {key: value.clone() for key, value in state.items()}
    embed, current = torch.randn(2, 8), torch.randn(2, 4, 8)
    post, prior = model.obs_step(state, torch.ones(2, 3), embed,
                                 torch.tensor([True, False]), sample=False,
                                 memory_tokens=current)
    fresh_post, fresh_prior = model.obs_step(None, None, embed[:1], torch.ones(1),
                                             sample=False, memory_tokens=current[:1])
    for key in post:
        # Batched and single-row linear algebra may differ in the final float bit.
        torch.testing.assert_close(post[key][:1], fresh_post[key])
        torch.testing.assert_close(prior[key][:1], fresh_prior[key])
    assert not prior["mem_valid"][0].any()
    assert prior["mem_valid"][1].sum().item() == 4
    assert_same_state(state, saved)


def test_memory_fifo_has_fixed_budget_and_only_real_observations_write():
    model = make_rssm()
    state = None
    for value in (1., 2., 3.):
        state, _ = model.obs_step(
            state, torch.zeros(1, 3), torch.zeros(1, 8),
            torch.tensor([value == 1.]), sample=False,
            memory_tokens=torch.full((1, 4, 8), value),
        )
    assert state["mem_tokens"].shape == (1, 8, 8)
    assert state["mem_valid"].all()
    assert torch.equal(state["mem_tokens"][:, :4], torch.full((1, 4, 8), 2.))
    assert torch.equal(state["mem_tokens"][:, 4:], torch.full((1, 4, 8), 3.))
    assert torch.equal(state["mem_age"], torch.tensor([[1.] * 4 + [0.] * 4]))


def test_start_only_does_not_read_changed_memory_after_first_prediction():
    model = make_rssm(read_mode="start_only")
    state, _ = model.obs_step(None, None, torch.zeros(1, 8), torch.ones(1),
                              sample=False, memory_tokens=torch.randn(1, 4, 8))
    step = model.img_step(state, torch.tensor([[1., 0., 0.]]), sample=False)
    changed = {key: value.clone() for key, value in step.items()}
    changed["mem_tokens"] += 17
    following = model.img_step(step, torch.tensor([[1., 0., 0.]]), sample=False)
    following_changed = model.img_step(changed, torch.tensor([[1., 0., 0.]]), sample=False)
    for key in ("deter", "stoch", "logit"):
        assert torch.equal(following[key], following_changed[key]), key


def test_attention_and_joint_dynamics_receive_multistep_prediction_gradient():
    model = make_rssm()
    state, _ = model.obs_step(None, None, torch.randn(2, 8), torch.ones(2),
                              sample=False, memory_tokens=torch.randn(2, 4, 8))
    state = {key: value.detach() for key, value in state.items()}
    for _ in range(3):
        state = model.img_step(state, torch.tensor([[1., 0., 0.]] * 2), sample=False)
    state["logit"].square().mean().backward()
    for module in (model.s, model.z, model.query, model.attention, model.read_out):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and any(gradient.abs().sum() > 0 for gradient in grads)
        assert all(torch.isfinite(gradient).all() for gradient in grads)


def test_burnin_detaches_history_and_invalid_padding_is_not_observed():
    model = make_rssm()
    embed = torch.randn(1, 5, 8, requires_grad=True)
    tokens = torch.randn(1, 5, 4, 8, requires_grad=True)
    actions = torch.tensor([[[1., 0., 0.]] * 5])
    first = torch.tensor([[False, True, False, False, False]])
    valid = torch.tensor([[False, True, True, True, True]])
    learning = torch.tensor([[False, False, False, True, True]])
    post, _ = model.observe(embed, actions, first, memory_tokens=tokens,
                             sequence_valid=valid, loss_mask=learning)
    assert not post["mem_valid"][:, 0].any()
    assert post["mem_valid"][:, 1].sum().item() == 4
    post["deter"][:, -1].square().sum().backward()
    assert embed.grad is not None and embed.grad[:, :3].abs().sum() == 0
    assert tokens.grad is not None and tokens.grad[:, :3].abs().sum() == 0
    assert tokens.grad[:, 3].abs().sum() > 0


def test_prefix_masks_include_terminal_reward_but_exclude_later_or_reset_rows():
    data = {
        "is_first": torch.tensor([[True, False, False, False], [True, False, True, False]]),
        "is_terminal": torch.tensor([[False, False, True, False], [False] * 4]),
        "is_last": torch.tensor([[False, False, True, False], [False] * 4]),
        "sequence_valid": torch.tensor([[True, True, True, False], [True] * 4]),
    }
    starts = torch.tensor([[0, 0], [0, 1], [1, 0], [0, 2]])
    actual = EvidenceWorldModel._valid_prefixes(data, starts, [1, 2, 3])
    assert torch.equal(actual, torch.tensor([
        [True, True, False], [True, False, False],
        [True, False, False], [False, False, False],
    ]))


def test_rc_uses_incoming_actions_raw_task_reward_and_terminal_endpoint():
    # A controlled transition/reader makes an off-by-one or reward-route error
    # visible numerically instead of hiding it behind an untrained neural head.
    options = dict(fc_weight=0., rc_weight=1., horizons=[2, 3],
                   horizon_weights=[1., 1.], starts_per_sequence=4,
                   mc_samples=2, freeze_rc_reward_head=False)
    fake = SimpleNamespace(
        options=options, _config=SimpleNamespace(discount=.5),
        dynamics=SimpleNamespace(
            img_step=lambda state, action: {"deter": action[:, :1]},
            get_feat=lambda state: state["deter"],
        ),
        heads={"task_reward": lambda features: SimpleNamespace(mode=lambda: features)},
    )
    fake._valid_prefixes = EvidenceWorldModel._valid_prefixes
    fake._select_starts = lambda data: EvidenceWorldModel._select_starts(fake, data)
    data = {
        "action": torch.tensor([[[99.], [2.], [4.], [999.]]]),
        "task_reward": torch.tensor([[[321.], [2.], [4.], [888.]]]),
        "clip_reward": torch.tensor([[[999.], [777.], [666.], [555.]]]),
        "is_first": torch.tensor([[True, False, False, False]]),
        "is_terminal": torch.tensor([[False, False, True, False]]),
        "is_last": torch.tensor([[False, False, True, False]]),
        "sequence_valid": torch.tensor([[True, True, True, False]]),
        "loss_mask": torch.tensor([[True, False, False, False]]),
    }
    post = {"deter": torch.zeros(1, 4, 1)}
    losses, metrics = EvidenceWorldModel._auxiliary_losses(fake, post, data)
    assert losses["rc"].item() == 0 and metrics["rc_abs_error_numerator"].item() == 0
    assert metrics["rc_prefixes"].item() == 1
    # Reward 4 at the terminal endpoint is included: removing it changes target
    # by gamma * 4 = 2, hence squared RC error becomes 4.
    changed = dict(data, task_reward=data["task_reward"].clone())
    changed["task_reward"][:, 2] = 0
    changed_loss, _ = EvidenceWorldModel._auxiliary_losses(fake, post, changed)
    assert changed_loss["rc"].item() == 4


def test_zero_auxiliary_weights_skip_paths_and_preserve_rng(tmp_path, monkeypatch):
    config = small_config()
    config["evidence"].update(fc_weight=0., rc_weight=0.)
    method = EvidenceDreamer(config, tmp_path)
    def prohibited(*args, **kwargs):
        raise AssertionError("Disabled FC/RC must not sample starts, read targets, or roll out")
    monkeypatch.setattr(method.wm, "_select_starts", prohibited)
    monkeypatch.setattr(method.wm, "_teacher", prohibited)
    monkeypatch.setattr(method.wm.dynamics, "img_step", prohibited)
    before = torch.get_rng_state().clone()
    losses, metrics = method.wm._auxiliary_losses({"deter": torch.zeros(1, 1, 16)}, {})
    assert torch.equal(before, torch.get_rng_state())
    assert all(value.item() == 0 for value in losses.values())
    assert metrics["fc_prefixes"].item() == metrics["rc_prefixes"].item() == 0


@pytest.mark.parametrize("ablation,field", [("evidence_no_fc", "fc_weight"),
                                           ("evidence_no_rc", "rc_weight")])
def test_named_zero_weight_ablation_matches_manual_zero_config(ablation, field):
    named = small_config(ablations=[ablation])
    manual = small_config()
    manual["evidence"][field] = 0.
    assert named["evidence"] == manual["evidence"]
    # The remaining architecture and objectives are deliberately still present.
    assert named["evidence"]["memory_enabled"]
    assert named["evidence"]["posterior_heat_weight"] > 0
    assert named["dreamer"] == manual["dreamer"]


def test_replay_keeps_clean_task_and_clip_incoming_labels():
    config = small_config()
    replay = EvidenceReplay(config)
    env = SyntheticEnv(config, needs_heatmap=True)
    first = env.reset()
    first.update(task_reward=99., intrinsic=77., clip_reward=55.)
    replay.begin(first)
    current = env.step(3)
    current.update(task_reward=2., clip_reward=.75, intrinsic=88., reward=123.,
                   is_last=True, is_terminal=True, done=True, discount=0.0)
    replay.add(3, current)
    replay.finish()
    episode = next(iter(replay.episodes.values()))
    np.testing.assert_array_equal(episode["task_reward"], [0., 2.])
    np.testing.assert_array_equal(episode["clip_reward"], [0., .75])
    np.testing.assert_array_equal(episode["reward"], [0., 2.75])
    assert not episode["action"][0].any()
    assert episode["action"][1].argmax() == 3
    np.testing.assert_array_equal(episode["image"][1], current["image"])
    assert episode["is_terminal"][-1] and episode["discount"][-1] == 0
    assert episode["heatmap_valid"].all()


def test_replay_contiguous_burnin_padding_and_true_episode_boundaries():
    config = small_config()
    config.update(batch_size=12, batch_length=8)
    replay = EvidenceReplay(config)
    for episode_id, count in ((30, 3), (90, 2)):
        first = dict(image=np.full((64, 64, 3), episode_id, np.uint8),
                     is_first=True, is_last=False, is_terminal=False,
                     task_reward=0., clip_reward=0., heatmap_valid=False)
        replay.begin(first)
        for step in range(1, count + 1):
            obs = dict(first, image=np.full((64, 64, 3), episode_id + step, np.uint8),
                       is_first=False, is_last=step == count,
                       task_reward=float(step), clip_reward=.01 * step)
            replay.add(step, obs)
        replay.finish()
    batch = replay.sample()
    assert batch["image"].shape == (12, 12, 64, 64, 3)
    assert not batch["loss_mask"][:, :4].any()
    assert not (batch["loss_mask"] & ~batch["sequence_valid"]).any()
    assert batch["loss_mask"][:, 4].all()
    for row in range(12):
        valid = batch["sequence_valid"][row]
        positions = np.flatnonzero(valid)
        np.testing.assert_array_equal(positions, np.arange(positions[0], positions[-1] + 1))
        images = batch["image"][row, valid, 0, 0, 0].astype(int)
        np.testing.assert_array_equal(np.diff(images), np.ones(len(images) - 1))
        assert (images < 50).all() or (images > 80).all()
        # A real reset can appear in burn-in, but it is never invented at a sample boundary.
        first = batch["is_first"][row, valid]
        assert first.sum() <= 1 and not batch["is_first"][row, ~valid].any()
        for value, reset in zip(images, first):
            assert bool(reset) == (value in (30, 90))
        assert batch["is_last"][row, positions[-1]]
        assert not batch["is_last"][row, positions[:-1]].any()
        assert not batch["is_terminal"][row, ~valid].any()


def test_replay_checkpoint_has_independent_rng_and_preserves_partial_episode():
    config = small_config()
    replay = EvidenceReplay(config)
    env = SyntheticEnv(config, needs_heatmap=True)
    replay.begin(env.reset())
    replay.add(1, env.step(1))
    saved = replay.state_dict()
    replay.add(1, env.step(1))
    clone = EvidenceReplay(config)
    clone.load_state_dict(saved)
    assert clone.active is None and clone.steps() == 1
    assert EvidenceReplay._length(saved["episodes"][replay.active]) == 2
    replay.load_state_dict(saved)
    expected = replay.sample()
    np.random.rand(100)
    random.random()
    torch.rand(30)
    actual = clone.sample()
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    actual["image"][:] = 255
    assert saved["episodes"]["1"][0]["image"][0, 0, 0] == 0


def test_policy_uses_rgb_and_ignores_arbitrary_teacher_or_future_labels(tmp_path):
    config, method, env = build(tmp_path)
    obs = env.reset()
    changed = dict(obs, heatmap=np.full((64, 64), 255, np.uint8),
                   heatmap_valid=False, task_reward=900., clip_reward=-42.,
                   future_image=np.full((64, 64, 3), 255, np.uint8))
    rng = torch.get_rng_state().clone()
    action_a, carry_a, extras_a = method.act(obs, None, training=False)
    torch.set_rng_state(rng)
    action_b, carry_b, extras_b = method.act(changed, None, training=False)
    assert action_a == action_b and not extras_a and not extras_b
    assert_same_state(carry_a[0], carry_b[0])
    assert torch.equal(carry_a[1], carry_b[1])
    _, carry, _ = method.act(obs, None, random_action=3)
    assert carry[1].argmax(-1).item() == 3
    assert method.wm.encoder.cnn_shapes == {"image": (64, 64, 3)}
    assert not method.wm.encoder.mlp_shapes
    assert list(method.wm.heads["decoder"].cnn_shapes) == ["image"]


def test_observation_scan_does_not_read_future_rgb_or_teacher_into_earlier_memory(tmp_path):
    _, method, _ = build(tmp_path)
    episode = next(iter(method.replay.episodes.values()))
    batch = {key: value[None].copy() for key, value in episode.items()}
    changed = copy.deepcopy(batch)
    changed["image"][:, 4:] = 255
    changed["heatmap"][:, 4:] = 0
    changed["task_reward"][:, 4:] = 999
    def observe(value):
        data = method.wm.preprocess(value)
        with torch.no_grad():
            tokens, _ = method.wm.writer(data["image"])
            return method.wm.dynamics.observe(method.wm.encoder(data), data["action"],
                                              data["is_first"], memory_tokens=tokens)
    rng = torch.get_rng_state().clone()
    post_a, prior_a = observe(batch)
    torch.set_rng_state(rng)
    post_b, prior_b = observe(changed)
    for key in post_a:
        assert torch.equal(post_a[key][:, :4], post_b[key][:, :4]), key
        assert torch.equal(prior_a[key][:, :4], prior_b[key][:, :4]), key
    assert not torch.equal(post_a["mem_tokens"][:, 4:], post_b["mem_tokens"][:, 4:])


def test_future_heatmap_readout_uses_dynamic_s_without_raw_memory_or_z(tmp_path):
    method = EvidenceDreamer(small_config(), tmp_path)
    state = method.wm.dynamics.initial(1)
    changed = {key: value.clone() for key, value in state.items()}
    changed["deter"][:, method.native.dyn_deter // 2:] += 100
    changed["stoch"][:, method.native.dyn_stoch // 2:] = 7
    changed["mem_tokens"][:] = 999
    changed["mem_valid"][:] = True
    assert torch.equal(method.wm.heatmap_logits(state), method.wm.heatmap_logits(changed))


def test_world_model_actor_critic_update_and_optimizer_registration(tmp_path):
    _, method, _ = build(tmp_path)
    original = {name: parameter.detach().clone() for name, parameter in method.wm.named_parameters()}
    actor = [parameter.detach().clone() for parameter in method.behavior.actor.parameters()]
    critic = [parameter.detach().clone() for parameter in method.behavior.value.parameters()]
    metrics = method.update()
    assert_finite(metrics)
    assert metrics["actor_valid_starts"] > 0 and not metrics["actor_skipped"]
    for prefix in ("dynamics.s.", "dynamics.z.", "writer."):
        assert any(not torch.equal(original[name], parameter)
                   for name, parameter in method.wm.named_parameters() if name.startswith(prefix)), prefix
    assert any(not torch.equal(saved, parameter) for saved, parameter in zip(actor, method.behavior.actor.parameters()))
    assert any(not torch.equal(saved, parameter) for saved, parameter in zip(critic, method.behavior.value.parameters()))
    optimized = {id(parameter) for group in method.wm._model_opt._opt.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in method.wm.parameters()}


def test_rc_gradients_train_joint_transition_and_reading_without_reward_head_updates(tmp_path):
    config, method, _ = build(tmp_path)
    method.wm.options.update(fc_weight=0., rc_weight=1., horizons=[1, 2],
                             horizon_weights=[1., 1.], starts_per_sequence=4)
    episode = next(iter(method.replay.episodes.values()))
    batch = {key: value[None].copy() for key, value in episode.items()}
    batch["sequence_valid"] = np.ones_like(batch["is_first"])
    batch["loss_mask"] = np.ones_like(batch["is_first"])
    data = method.wm.preprocess(batch)
    with torch.no_grad():
        tokens, _ = method.wm.writer(data["image"])
        post, _ = method.wm.dynamics.observe(
            method.wm.encoder(data), data["action"], data["is_first"],
            memory_tokens=tokens, sequence_valid=data["sequence_valid"],
        )
        # The native zero-output initialization makes input derivatives zero;
        # use a calibrated nonconstant head to test the intended later-training path.
        method.wm.heads["task_reward"].mean_layer.weight.normal_(std=.02)
    method.wm.requires_grad_(True)
    method.wm.zero_grad(set_to_none=True)
    losses, metrics = method.wm._auxiliary_losses(post, data)
    assert metrics["rc_prefixes"] > 0 and losses["rc"].requires_grad
    losses["rc"].backward()
    for module in (method.wm.dynamics.s, method.wm.dynamics.z,
                   method.wm.dynamics.query, method.wm.dynamics.attention):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and any(gradient.abs().sum() > 0 for gradient in grads)
        assert all(torch.isfinite(gradient).all() for gradient in grads)
    assert all(parameter.grad is None for parameter in method.wm.heads["task_reward"].parameters())
    assert all(parameter.requires_grad for parameter in method.wm.heads["task_reward"].parameters())
    assert all(parameter.grad is None for parameter in method.wm.writer.parameters())


def test_independent_seeds_change_model_and_same_seed_repeats_it(tmp_path):
    config = small_config()
    first = EvidenceDreamer(config, tmp_path)
    match = EvidenceDreamer(copy.deepcopy(config), tmp_path)
    assert_same_state(first.wm.state_dict(), match.wm.state_dict())
    assert_same_state(first.behavior.actor.state_dict(), match.behavior.actor.state_dict())
    different_config = copy.deepcopy(config)
    different_config["seed"] += 1
    different = EvidenceDreamer(different_config, tmp_path)
    assert any(not torch.equal(parameter, different.wm.state_dict()[key])
               for key, parameter in first.wm.state_dict().items())
    assert any(not torch.equal(parameter, different.wm.dynamics.state_dict()[key])
               for key, parameter in first.wm.dynamics.state_dict().items())


def test_checkpoint_restores_exact_next_batch_and_update(tmp_path):
    config, method, _ = build(tmp_path)
    method.update()
    saved = method.state_dict()
    clone = EvidenceDreamer(config, tmp_path)
    clone.load_state_dict(saved)
    expected, actual = method.replay.sample(), clone.replay.sample()
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    method.load_state_dict(saved)
    metrics_one = method.update()
    model_one = copy.deepcopy(method.wm.state_dict())
    behavior_one = copy.deepcopy(method.behavior.state_dict())
    clone.load_state_dict(saved)
    metrics_two = clone.update()
    assert_same_state(model_one, clone.wm.state_dict())
    assert_same_state(behavior_one, clone.behavior.state_dict())
    assert metrics_one.keys() == metrics_two.keys()
    for key in metrics_one:
        np.testing.assert_allclose(metrics_one[key], metrics_two[key], atol=0, rtol=0)
    assert clone.updates == method.updates == 2


def test_evaluation_restores_training_rng_modes_and_replay(tmp_path):
    config, method, env = build(tmp_path)
    replay_before = copy.deepcopy(method.replay.state_dict())
    model_before = copy.deepcopy(method.wm.state_dict())
    torch_rng, np_rng, py_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    modes = [module.training for module in method.behavior.modules()]
    with method.evaluation(872):
        obs, carry = env.reset(), None
        for _ in range(4):
            action, carry, extras = method.act(obs, carry, training=False)
            assert 0 <= action < config["num_actions"] and not extras
            obs = env.step(action)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert py_rng == random.getstate()
    np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
    assert modes == [module.training for module in method.behavior.modules()]
    assert_same_state(model_before, method.wm.state_dict())
    after = method.replay.state_dict()
    assert replay_before["counter"] == after["counter"]
    np.testing.assert_array_equal(replay_before["rng"][1], after["rng"][1])
    for episode in replay_before["episodes"]:
        for key in replay_before["episodes"][episode]:
            np.testing.assert_array_equal(replay_before["episodes"][episode][key], after["episodes"][episode][key])




def test_replay_respects_clean_discount_and_original_terminal_flags():
    config = small_config()
    env = SyntheticEnv(config)
    replay = EvidenceReplay(config)
    replay.begin(env.reset())
    current = env.step(3)
    current.update(discount=0.0, is_last=True, is_terminal=False, done=True)
    replay.add(3, current)
    replay.finish()
    episode = next(iter(replay.episodes.values()))
    assert episode['discount'][-1] == 0.0
    assert episode['is_last'][-1] and not episode['is_terminal'][-1]
    assert episode['action'][-1].argmax() == 3


def test_actor_does_not_start_from_clean_last_but_nonterminal_rows(tmp_path, monkeypatch):
    _, method, _ = build(tmp_path)
    batch = method.replay.sample()
    batch['loss_mask'][:] = False
    endpoints = []
    for row in range(batch['sequence_valid'].shape[0]):
        endpoint = np.flatnonzero(batch['sequence_valid'][row])[-1]
        endpoints.append((row, endpoint))
        batch['loss_mask'][row, endpoint] = True
        batch['is_last'][row, endpoint] = True
        batch['is_terminal'][row, endpoint] = False
        batch['discount'][row, endpoint] = 0.0
    monkeypatch.setattr(method.replay, 'sample', lambda: batch)
    def prohibited(*args, **kwargs):
        raise AssertionError('An ended Clean episode cannot supply actor imagination starts')
    monkeypatch.setattr(method.behavior, '_train', prohibited)
    metrics = method.update()
    assert metrics['actor_skipped'] == 1 and metrics['actor_valid_starts'] == 0
    assert_finite(metrics)
    for row, endpoint in endpoints:
        assert batch['is_last'][row, endpoint]
        assert not batch['is_terminal'][row, endpoint]


@pytest.mark.parametrize('discount', [float('nan'), -0.1, 1.1, [0., 1.]])
def test_replay_rejects_invalid_clean_discounts(discount):
    config = small_config()
    env = SyntheticEnv(config)
    replay = EvidenceReplay(config)
    replay.begin(env.reset())
    obs = env.step(2)
    obs['discount'] = discount
    with pytest.raises(ValueError, match='discount'):
        replay.add(2, obs)
