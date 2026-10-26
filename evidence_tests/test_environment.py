"""Real Clean wrappers around fake physics and frozen teacher interfaces."""
import copy
import importlib
import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
# The project targets Python 3.9 + Gym 0.21. Gym's plugin discovery expects the
# old entry_points mapping API; adapt that metadata-only import boundary when
# running these source-contract tests under the local Python 3.12 verifier.
_entry_points = importlib.metadata.entry_points
_points = _entry_points()
if not hasattr(_points, 'get'):
    _legacy_points = {}
    for _point in _points:
        _legacy_points.setdefault(_point.group, []).append(_point)
    importlib.metadata.entry_points = lambda: _legacy_points
try:
    import gym
finally:
    importlib.metadata.entry_points = _entry_points
import numpy as np
import pytest
import torch

from clean_evidence.environment import CleanEvidenceEnv
from clean_evidence.semantic_teacher import make_heatmap_only_wrapper


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    # Each fixture imports the actual Clean source against its own tiny
    # MineDojo/teacher modules. No installed MineDojo or teacher is modified.
    for name in list(sys.modules):
        if name == 'envs' or name.startswith('envs.'):
            monkeypatch.delitem(sys.modules, name)
    state = SimpleNamespace(constructs=[], sims=[], clips=[], teachers=[], masks=0)

    def module(name, **values):
        value = ModuleType(name)
        value.__dict__.update(values)
        value.__path__ = []
        monkeypatch.setitem(sys.modules, name, value)
        return value

    class FakeSim(gym.Env):
        action_space = gym.spaces.Dict({})
        observation_space = gym.spaces.Dict({})

        def __init__(self):
            self.full_resets = 0
            self.fast_resets = 0
            self.actions = []
            self.closed = 0
            self.quantity = 0
            self.t = 0
            state.sims.append(self)

        def observation(self):
            rgb = np.full((3, 160, 256), 37, np.uint8)
            rgb[:, 140:, 90:160] = 244
            return {'rgb': rgb, 'inventory': {'name': ['log'], 'quantity': [self.quantity]},
                    'life_stats': {'life': np.array(20)}}

        def reset(self):
            self.full_resets += 1
            self.quantity = 0
            self.t = 0
            return self.observation()

        def step(self, action):
            self.actions.append(copy.deepcopy(action))
            self.t += 1
            acquired = bool(action['attack']) and not self.quantity
            self.quantity = int(self.quantity or acquired)
            return self.observation(), float(acquired), False, {}

        def close(self):
            self.closed += 1

    class FakeFastReset(gym.Wrapper):
        def __init__(self, env, random_teleport_range=200):
            super().__init__(env)
            self.random_teleport_range = random_teleport_range

        def reset(self):
            self.env.fast_resets += 1
            return self.env.reset()

    def make_task(cls, **kwargs):
        state.constructs.append((cls, copy.deepcopy(kwargs)))
        return gym.Wrapper(FakeSim())

    class FakeClip:
        def __init__(self, **kwargs):
            state.clips.append(kwargs)

        def get_logits(self, observation, prompts, current):
            return torch.tensor([30.] * len(prompts)), current

    class FakeConcentration:
        def __init__(self, **kwargs):
            state.teachers.append(kwargs)

        def _generate_mask(self, observation, prompts):
            np.random.rand(9)
            state.masks += 1
            return np.full((len(prompts), 160, 256), .75, np.float32)

        def get_reward(self, *args):
            return .8, 0., 0.

        def generate_zoom_in_frame(self):
            return np.zeros((160, 256, 3), np.uint8), False

        def get_heatmap(self, **kwargs):
            return np.full((160, 256, 1), .75, np.float32)

    module('minedojo')
    module('minedojo.tasks', MetaTaskBase=gym.Env, _meta_task_make=make_task,
           _parse_inventory_dict=lambda value: value, ALL_TASKS_SPECS={})
    module('minedojo.sim', MineDojoSim=FakeSim)
    module('minedojo.sim.wrappers')
    module('minedojo.sim.wrappers.fast_reset', FastResetWrapper=FakeFastReset)
    module('minedojo.sim.mc_meta')
    module('minedojo.sim.mc_meta.mc', **{name: {} for name in [
        'ALL_ITEMS', 'ALL_PERSONAL_CRAFTING_ITEMS', 'ALL_CRAFTING_TABLE_ITEMS',
        'ALL_SMELTING_ITEMS', 'CRAFTING_RECIPES_BY_OUTPUT', 'SMELTING_RECIPES_BY_OUTPUT']})
    module('envs.tasks.base.clip_reward', ClipReward=FakeClip)
    module('envs.tasks.base.concentration_reward', ConcentrationReward=FakeConcentration)
    state.tasks = importlib.import_module('envs.tasks')
    state.original_specs = copy.deepcopy(state.tasks.CUTOM_TASK_SPECS)
    state.assets = tmp_path / 'assets'
    state.assets.mkdir()
    state.config = {'task': 'minedojo_harvest_log_in_plains', 'assets_root': str(state.assets),
                    'device': 'cpu', 'env_device': 'cpu', 'clean': {}}
    state.run_dir = tmp_path / 'run'
    return state


def chain(env):
    result = []
    while env is not None:
        result.append(type(env).__name__)
        env = getattr(env, 'env', None)
    return result


def test_evidence_uses_clean_actions_binary_reward_hud_and_terminal_contract(clean_runtime):
    state = clean_runtime
    env = CleanEvidenceEnv(state.config, run_dir=state.run_dir)
    try:
        assert env.num_actions == 12 and env.max_episode_steps == 1000
        layers = chain(env.env)
        assert layers[:4] == ['UUID', 'SelectAction', 'OneHotAction', 'MinedojoLSImagineWrapper']
        assert 'MinedojoScreenshotWrapper' in layers
        assert 'HeatmapOnlyWrapper' in layers and 'ConcentrationWrapper' not in layers
        assert 'MinedojoSemifastResetWrapper' in layers
        assert 'MinedojoRewardWrapper' not in layers
        initial = env.reset()
        assert initial['task_reward'] == initial['clip_reward'] == initial['intrinsic'] == 0
        assert initial['heatmap_valid'] and initial['discount'] == 1 and initial['is_first']
        assert initial['image'].shape == (64, 64, 3)
        assert initial['heatmap'].shape == (64, 64) and initial['heatmap'][0, 0] == 191
        image = state.sims[0].observation()['rgb'].transpose(1, 2, 0)
        mask = cv2.imread(str(ROOT / 'envs/tasks/base/HUD_mask.png'), cv2.IMREAD_GRAYSCALE)
        expected = cv2.resize(cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA), (64, 64))
        np.testing.assert_array_equal(initial['image'], expected)
        current = env.step(1)  # Clean BASIC_ACTIONS attack
        assert current['task_reward'] == current['reward'] == 1.0
        assert current['success'] and current['done'] and current['is_last']
        assert not current['is_terminal'] and current['discount'] == 0
        assert current['first_success_step'] == 0  # preserve Clean's counter
        assert current['clip_reward'] == current['intrinsic'] > 0
        assert not current['is_zoomed'] and not current['jump']
        assert int(state.sims[0].actions[-1]['attack']) == 1
        assert state.constructs[0][1]['image_size'] == (160, 256)
        assert 'seed' not in state.constructs[0][1] and 'world_seed' not in state.constructs[0][1]
        assert state.tasks.CUTOM_TASK_SPECS == state.original_specs
        assert state.clips[0]['ckpt'] == str(state.assets / 'weights/mineclip_attn.pth')
        assert state.teachers[0]['unet_checkpoint_dir'].endswith('harvest_log_in_plains')
    finally:
        env.close()
        env.close()
    assert state.sims[0].closed == 1


def test_eval_keeps_same_task_fast_reset_and_does_not_construct_unet(clean_runtime):
    state = clean_runtime
    env = CleanEvidenceEnv(state.config, mode='eval', run_dir=state.run_dir)
    try:
        for _ in range(6):
            packet = env.reset()
            assert not packet['heatmap_valid']
        assert state.teachers == [] and state.masks == 0
        assert len(state.constructs) == 1
        # Eval uses the same fast_reset=5 task, not WM's per-episode full reset.
        assert state.sims[0].fast_resets == 5
        assert 'MinedojoSemifastResetWrapper' in chain(env.env)
    finally:
        env.close()


def test_no_semantic_training_keeps_clip_and_clean_reset_without_unet(clean_runtime):
    state = clean_runtime
    config = dict(state.config, needs_heatmap=False)
    env = CleanEvidenceEnv(config, mode='train', run_dir=state.run_dir)
    try:
        initial = env.reset()
        assert not initial['heatmap_valid']
        assert 'HeatmapOnlyWrapper' not in chain(env.env)
        assert 'ConcentrationWrapper' not in chain(env.env)
        assert 'ClipWrapper' in chain(env.env)
        assert 'MinedojoSemifastResetWrapper' in chain(env.env)
        current = env.step(1)
        assert current['task_reward'] == 1 and current['clip_reward'] > 0
        assert not current['heatmap_valid'] and state.teachers == [] and state.masks == 0
    finally:
        env.close()


def test_explicit_test_task_retains_its_own_zero_fast_reset(clean_runtime):
    state = clean_runtime
    config = dict(state.config, task='minedojo_test_harvest_log_in_plains')
    env = CleanEvidenceEnv(config, mode='eval', run_dir=state.run_dir)
    try:
        env.reset()
        env.reset()
        assert state.sims[0].fast_resets == 0
        assert state.sims[0].full_resets == 2
    finally:
        env.close()


def test_default_clean_ls_branch_still_constructs_concentration_shaping(clean_runtime):
    state = clean_runtime
    env = state.tasks.make('harvest_log_in_plains', log_dir=str(state.run_dir))
    try:
        assert 'ConcentrationWrapper' in chain(env)
        assert 'HeatmapOnlyWrapper' not in chain(env)
        obs = env.reset()
        assert obs['intrinsic'] > 0  # unchanged LS Gaussian shaping on reset
        assert state.clips == [{}]  # default teacher constructor unchanged
        assert 'ckpt' not in state.teachers[0]
    finally:
        env.close()


def test_evidence_physical_actions_sticky_jump_and_pitch_match_original_clean(clean_runtime):
    state = clean_runtime
    evidence = CleanEvidenceEnv(state.config, run_dir=state.run_dir)
    original = state.tasks.make('harvest_log_in_plains', log_dir=str(state.run_dir / 'ls'))
    try:
        evidence.reset()
        original.reset()
        actions = [10, 0] + [2] * 9 + [3, 4, 5, 6, 7, 8, 9, 11, 1]
        for action in actions:
            evidence.step(action)
            original.step(action)
        left, right = state.sims[0].actions, state.sims[1].actions
        assert len(left) == len(right) == len(actions)
        for actual, expected in zip(left, right):
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
        assert int(left[1]['jump']) == int(left[1]['forward']) == 1
        assert np.asarray(left[10]['camera'])[0] == 0  # source pitch cap
        with pytest.raises(ValueError, match='12-action'):
            evidence.step(12)
    finally:
        evidence.close()
        original.close()


def test_timeout_uses_clean_task_yaml_and_preserves_native_truncation_fields(clean_runtime):
    state = clean_runtime
    state.tasks.CUTOM_TASK_SPECS['harvest_log_in_plains']['terminal_specs']['max_steps'] = 3
    config = dict(state.config, max_episode_steps=9999)
    env = CleanEvidenceEnv(config, mode='eval', run_dir=state.run_dir)
    try:
        assert env.max_episode_steps == 3
        env.reset()
        assert not env.step(0)['done']
        assert not env.step(0)['done']
        last = env.step(0)
        assert last['done'] and last['is_last']
        assert not last['is_terminal'] and last['discount'] == 0
        assert last['task_reward'] == 0 and not last['success']
    finally:
        env.close()


def test_semantic_target_preserves_random_stream_and_rejects_invalid_masks():
    class Base(gym.Env):
        observation_space = gym.spaces.Dict({})
        action_space = gym.spaces.Discrete(12)
        def reset(self):
            return {'intrinsic': .3}
    class Teacher:
        def _generate_mask(self, obs, prompts):
            np.random.rand(3)
            return np.ones((1, 3, 4), np.float32)
    env = make_heatmap_only_wrapper(Base(), Teacher(), ['Cut a tree'])
    before = np.random.get_state()
    obs = env.reset()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:] and obs['intrinsic'] == .3
    bad = SimpleNamespace(_generate_mask=lambda obs, prompts: np.full((1, 2, 3), np.nan))
    with pytest.raises(ValueError, match='finite'):
        make_heatmap_only_wrapper(Base(), bad, ['task']).reset()
