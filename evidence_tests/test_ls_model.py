"""Real LS-Imagine learner checks on tiny synthetic replay; no Minecraft claim."""
import copy

import numpy as np
import pytest
import torch

# Existing Gym 0.21 metadata compatibility import for the Python 3.12 verifier.
from .test_environment import gym
from clean_ls.configuration import build_config, namespace
from expr import LS_Imagine
import tools


torch.set_num_threads(1)


class QuietLogger:
    step = 0

    def scalar(self, *args):
        pass

    def video(self, *args):
        pass

    def write(self, *args, **kwargs):
        pass


def tiny_ls(tmp_path, *, jump_prob=0.7):
    config = build_config(stage='smoke', logdir=tmp_path, overrides=[
        'device=cpu', 'env_device=cpu', 'clean.video_pred_log=false',
        f'clean.jump_prob={jump_prob}'])
    recipe = namespace(config)
    space = gym.spaces.Dict({
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'heatmap': gym.spaces.Box(0, 255, (64, 64, 1), dtype=np.uint8),
    })
    action = gym.spaces.Box(0, 1, (12,), dtype=np.float32)
    action.discrete = True
    tools.set_seed_everywhere(7)
    model = LS_Imagine(space, action, recipe, QuietLogger(), iter(())).to('cpu')
    model.requires_grad_(False)
    # Visualization sampling is unrelated to the learner; avoid random image I/O.
    model._task_behavior.save_state_sequence = lambda *a, **kw: None
    return model


def replay_batch(*, zoomed):
    batch, time = 2, 8
    rng = np.random.RandomState(12)
    data = {
        'image': rng.randint(0, 256, (batch, time, 64, 64, 3), dtype=np.uint8),
        'heatmap': rng.randint(0, 256, (batch, time, 64, 64), dtype=np.uint8),
        'zoomed_image': rng.randint(0, 256, (batch, time, 64, 64, 3), dtype=np.uint8),
        'heatmap_on_zoomed': rng.randint(0, 256, (batch, time, 64, 64), dtype=np.uint8),
        'action': np.eye(12, dtype=np.float32)[rng.randint(0, 12, (batch, time))],
    }
    for key in ('is_first', 'is_last', 'is_terminal', 'is_zoomed', 'is_calculated',
                'jump', 'reward', 'intrinsic', 'reward_on_zoomed', 'intrinsic_on_zoomed',
                'jumping_steps', 'accumulated_reward'):
        data[key] = np.zeros((batch, time), np.float32)
    data['is_first'][:, 0] = 1
    data['reward'][:, -1] = 1
    data['intrinsic'][:] = .1
    data['discount'] = np.ones((batch, time), np.float32)
    if zoomed:
        data['is_zoomed'][:, 2:4] = 1
        data['is_calculated'][:, 2] = 1  # both labeled and not-yet-labeled zooms
        data['jump'][:, 2:4] = 1
        data['jumping_steps'][:, 2] = 3
        data['accumulated_reward'][:, 2] = .5
        data['intrinsic_on_zoomed'][:, 2:4] = .3
    return data


@pytest.mark.parametrize('zoomed', [False, True])
def test_original_ls_updates_model_actor_critic_and_keeps_12_real_actions(tmp_path, zoomed):
    model = tiny_ls(tmp_path)
    data = replay_batch(zoomed=zoomed)
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in [('encoder', model._wm.encoder),
                                   ('actor', model._task_behavior.actor),
                                   ('value', model._task_behavior.value)]}
    zoom_heads_before = {name: [p.detach().clone() for p in model._wm.heads[name].parameters()]
                         for name in ('jumping_steps', 'accumulated_reward')}
    model._train(copy.deepcopy(data))
    for name, module in [('encoder', model._wm.encoder),
                         ('actor', model._task_behavior.actor),
                         ('value', model._task_behavior.value)]:
        assert any(not torch.equal(old, new)
                   for old, new in zip(before[name], module.parameters())), name
    for key, values in model._metrics.items():
        assert np.isfinite(np.asarray(values)).all(), key
    assert ('zoomed_jumping_steps_loss' in model._metrics) == zoomed
    assert ('zoomed_accumulated_reward_loss' in model._metrics) == zoomed
    for name, saved in zoom_heads_before.items():
        changed = any(not torch.equal(old, new)
                      for old, new in zip(saved, model._wm.heads[name].parameters()))
        assert changed == zoomed, name
    # The virtual zoom action belongs only to the world model.
    assert model._wm.dynamics._num_actions == 13
    observation = {k: v[:, 0] for k, v in data.items() if k != 'action'}
    for training in (False, True):
        output, state = model._policy(observation, None, training=training)
        assert output['action'].shape == (2, 12)
        assert torch.all(output['action'].sum(-1) == 1)
        assert torch.isfinite(output['logprob']).all()
        assert state[1].shape == (2, 12)


def test_original_ls_long_term_imagination_handles_positive_and_empty_branches(tmp_path):
    model = tiny_ls(tmp_path, jump_prob=1.0)
    data = replay_batch(zoomed=True)
    post, zoomed_post, _, _ = model._wm._train(copy.deepcopy(data))
    behavior = model._task_behavior
    reward = lambda f, s, a: model._wm.heads['reward'](f).mode()
    intrinsic = lambda f, s, a: model._wm.heads['intrinsic'](f).mode()
    jumping_steps = lambda f, s, a: model._wm.heads['jumping_steps'](f).mean().clamp_min(1).int()
    accumulated = lambda f, s, a: model._wm.heads['accumulated_reward'](f).mode()
    for probability in (0.0, 1.0):
        # Force only the route decision, retaining actual RSSM transitions and
        # actor/value gradients on both empty and nonempty long-term paths.
        jump = lambda s: torch.full((*s['deter'].shape[:-1], 1), probability)
        end = lambda s: torch.zeros((*s['deter'].shape[:-1], 1))
        feat, state, action, weights, metrics = behavior._train(
            copy.deepcopy(post), copy.deepcopy(zoomed_post), reward, intrinsic,
            jumping_steps, accumulated, jump, end)
        assert torch.isfinite(feat).all() and torch.isfinite(weights).all()
        assert action.shape[-1] == 12
        assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
        initial_starts = post['deter'].shape[0] * post['deter'].shape[1] + zoomed_post['deter'].shape[0]
        assert (feat.shape[1] > initial_starts) == bool(probability)
