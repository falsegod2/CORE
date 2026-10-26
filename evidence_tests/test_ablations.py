"""Named controls change the executed model, not only the experiment label."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from clean_evidence.ablations import ABLATIONS, requires_teacher
from clean_evidence.configuration import PROJECT_ROOT, build_config, resource_paths


def test_controls_keep_clean_protocol_and_have_disjoint_directories():
    configs = [build_config(ablation=name, seed=1) for name in ABLATIONS]
    full = configs[0]
    assert len({config['logdir'] for config in configs}) == len(ABLATIONS)
    for config in configs:
        assert config['protocol'] == full['protocol']
        assert config['dreamer'] == full['dreamer']
        assert config['checkpoint_every'] == 20000
        assert config['needs_heatmap'] == requires_teacher(config['evidence'])
        for key in ('num_actions', 'prefill', 'pretrain_updates', 'batch_size', 'batch_length',
                    'train_ratio', 'eval_every', 'eval_episodes', 'max_episode_steps'):
            assert config[key] == full[key], key
    smoke = build_config(ablation='full', stage='smoke', seed=1)
    assert smoke['logdir'] != full['logdir']


@pytest.mark.parametrize('name,key,value', [
    ('no_memory', 'memory_enabled', 'true'),
    ('no_memory_no_rc', 'memory_enabled', 'true'),
    ('no_memory_no_rc', 'rc_weight', '0.01'),
    ('no_memory_no_fc_rc', 'fc_weight', '0.05'),
    ('no_memory_no_fc_rc', 'rc_weight', '0.01'),
    ('no_memory_no_fc_rc', 'read_mode', 'each_step'),
    ('memory_start_only', 'read_mode', 'each_step'),
    ('no_fc', 'fc_weight', '0.05'),
    ('no_rc', 'rc_weight', '0.01'),
    ('no_fc_rc', 'rc_weight', '0.01'),
    ('no_semantic', 'local_heat_weight', '1.0'),
])
def test_control_switch_cannot_be_silently_reenabled(name, key, value):
    with pytest.raises(ValueError, match='requires evidence'):
        build_config(ablation=name, overrides=[f'evidence.{key}={value}'])


def test_no_semantic_does_not_require_unet_resource(tmp_path):
    config = build_config(ablation='no_semantic', assets_root=tmp_path)
    assert set(resource_paths(config)) == {'mineclip'}
    assert not config['needs_heatmap']
    assert config['evidence']['fc_weight'] > 0 and config['evidence']['fc_hidden_weight'] > 0
    assert config['evidence']['rc_weight'] > 0 and config['evidence']['clip_scale'] == 1.0


def test_public_listing_and_no_semantic_dry_run(tmp_path):
    command = [sys.executable, str(PROJECT_ROOT / 'run_evidence.py')]
    listed = subprocess.run(command + ['--list-ablations'], capture_output=True, text=True,
                            env=os.environ.copy(), check=True)
    assert [item['name'] for item in json.loads(listed.stdout)] == list(ABLATIONS)
    result = subprocess.run(command + ['--ablation', 'no_semantic', '--gpu', '1', '--seed', '3',
                                      '--assets-root', str(tmp_path), '--dry-run'],
                            capture_output=True, text=True, env=os.environ.copy(), check=True)
    plan = json.loads(result.stdout)
    assert plan['gpu'] == '1'
    assert plan['config']['ablation'] == 'no_semantic'
    assert not plan['config']['needs_heatmap']
    assert '/no_semantic/formal/' in plan['config']['logdir']


@pytest.mark.parametrize('name', list(ABLATIONS))
def test_each_named_control_updates_world_model_actor_critic_and_restores(tmp_path, name, monkeypatch):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from clean_evidence.methods.evidence_dreamer import EvidenceDreamer
    from evidence_tests.test_model_core import SyntheticEnv
    config = build_config(ablation=name, stage='smoke', seed=31,
                          overrides=['device=cpu', 'env_device=cpu'])
    method = EvidenceDreamer(config, tmp_path / name)
    env = SyntheticEnv(config, needs_heatmap=config['needs_heatmap'])
    for _ in range(3):
        previous = env.reset()
        if not config['needs_heatmap']:
            previous.pop('heatmap')
        method.begin_episode(previous)
        for _ in range(8):
            current = env.step(1)
            if not config['needs_heatmap']:
                current.pop('heatmap')
            method.record(previous, 1, current, {})
            previous = current
            if current['done']:
                break
        method.finish_episode()
    if name == 'no_semantic':
        def no_teacher(*args, **kwargs):
            raise AssertionError('no_semantic must never read a semantic target')
        monkeypatch.setattr(method.wm, '_teacher', no_teacher)
        assert method.wm.task_heatmap is None
    parameters = {key: value.detach().clone() for key, value in method.wm.named_parameters()}
    metrics = method.update()
    assert method.updates == 1 and metrics['actor_skipped'] == 0
    assert np.isfinite([value for value in metrics.values() if isinstance(value, (float, int))]).all()
    assert any(not torch.equal(parameters[key], value) for key, value in method.wm.named_parameters())
    if name in ('no_fc', 'no_fc_rc', 'no_memory_no_fc_rc'):
        assert metrics['fc_loss'] == metrics['fc_prefixes'] == 0
    if name in ('no_rc', 'no_fc_rc', 'no_memory_no_rc', 'no_memory_no_fc_rc'):
        assert metrics['rc_loss'] == metrics['rc_prefixes'] == 0
    if name in ('no_memory', 'no_memory_no_rc', 'no_memory_no_fc_rc'):
        assert not hasattr(method.wm.dynamics, 'attention')
        assert metrics['memory_fill'] == 0
        # The independent local semantic predictor remains trained, but the
        # memory token projection receives no gradient or optimizer state.
        for parameter in method.wm.writer.token_projection.parameters():
            assert parameter not in method.wm._model_opt._opt.state
        assert method.wm.writer.quadrant_embedding not in method.wm._model_opt._opt.state
        assert not torch.equal(parameters['writer.local_heat.weight'], method.wm.writer.local_heat.weight)
    if name == 'no_semantic':
        assert 'local_heat_loss' not in metrics and 'posterior_heat_loss' not in metrics
        assert metrics['future_heatmap_loss'] == 0 and metrics['semantic_rows'] == 0
    action, _, _ = method.act(env.reset(), None, training=False)
    assert 0 <= action < 12
    saved = method.state_dict()
    restored = EvidenceDreamer(config, tmp_path / (name + '_restored'))
    restored.load_state_dict(saved)
    assert restored.updates == 1 and restored.replay.steps() == method.replay.steps()
    for key, value in method.wm.state_dict().items():
        assert torch.equal(value, restored.wm.state_dict()[key]), key
    assert restored.parameter_report()['ablation'] == name


@pytest.mark.parametrize('name,fc', [('no_memory_no_rc', 0.05),
                                   ('no_memory_no_fc_rc', 0.0)])
def test_combination_controls_preserve_real_reward_and_semantic_supervision(name, fc):
    full = build_config()
    c = build_config(ablation=name)
    assert c['needs_heatmap'] and c['semantic_teacher_only']
    assert c['protocol'] == full['protocol']
    assert not c['evidence']['memory_enabled'] and c['evidence']['read_mode'] == 'none'
    assert c['evidence']['rc_weight'] == 0 and c['evidence']['fc_weight'] == fc
    for key in ('task_reward_weight', 'clip_reward_weight', 'clip_scale',
                'local_heat_weight', 'posterior_heat_weight'):
        assert c['evidence'][key] == full['evidence'][key] == 1.0


def test_start_only_skips_attention_after_first_imagined_transition(monkeypatch):
    torch = pytest.importorskip('torch')
    from evidence_tests.test_model_core import make_rssm
    model = make_rssm(read_mode='start_only')
    state, _ = model.obs_step(None, None, torch.zeros(1, 8), torch.ones(1),
                              sample=False, memory_tokens=torch.randn(1, 4, 8))
    action = torch.tensor([[1., 0., 0.]])
    first = model.img_step(state, action, sample=False)
    def prohibited(*args, **kwargs):
        raise AssertionError('later start-only transitions must skip attention')
    monkeypatch.setattr(model.attention, 'forward', prohibited)
    for _ in range(4):
        first = model.img_step(first, action, sample=False)


def test_all_controls_share_initial_common_world_actor_and_value_weights(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from clean_evidence.methods.evidence_dreamer import EvidenceDreamer
    def construct(name):
        config = build_config(ablation=name, stage='smoke', seed=31,
                              overrides=['device=cpu', 'env_device=cpu'])
        return EvidenceDreamer(config, tmp_path / name)
    # Behavior registers the world model, actor, value and slow value together.
    full = construct('full').behavior.state_dict()
    for name in ABLATIONS:
        actual = construct(name).behavior.state_dict()
        shared = set(actual) & set(full)
        assert any(key.startswith('actor.') for key in shared)
        assert any(key.startswith('value.') for key in shared)
        assert any(key.startswith('_world_model.') for key in shared)
        for key in shared:
            assert torch.equal(actual[key], full[key]), (name, key)
