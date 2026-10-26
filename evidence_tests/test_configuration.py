"""The public entry uses the uploaded Clean task contract, not the WM suite."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from clean_evidence.configuration import PROJECT_ROOT, build_config, task_specs, experiment_manifest


def test_full_evidence_recipe_uses_clean_runtime_defaults():
    config = build_config(seed=1)
    _, specs = task_specs(config['task'])
    assert config['max_episode_steps'] == specs['terminal_specs']['max_steps'] == 1000
    assert config['protocol']['fast_resets'] == specs['fast_reset'] == 5
    assert specs.get('reward_specs') is None
    assert specs['success_specs']['reward'] == 0
    assert config['protocol']['eval_action'] == 'mode'
    assert config['eval_every'] == 20000 and config['eval_episodes'] == 10
    assert config['prefill'] == 2500 and config['pretrain_updates'] == 100
    assert config['train_ratio'] == 16 and config['batch_size'] == 16 and config['batch_length'] == 32
    assert config['num_actions'] == 12
    assert config['dreamer']['overrides']['critic']['layers'] == 5
    for section in ('encoder', 'decoder'):
        assert config['dreamer']['overrides'][section]['cnn_keys'] == 'image'
        assert config['dreamer']['overrides'][section]['mlp_keys'] == '$^'
    assert config['evidence']['memory_enabled']
    assert config['evidence']['fc_weight'] == .05 and config['evidence']['rc_weight'] == .01


@pytest.mark.parametrize('override', ['protocol.eval_action=sample', 'protocol.fast_resets=0',
                                     'max_episode_steps=999', 'num_actions=13',
                                     'evidence.burnin=1', 'evidence.clip_scale=2',
                                     'dreamer.overrides.encoder.cnn_keys=heatmap',
                                     'semantic_teacher_only=false'])
def test_environment_and_rgb_contract_cannot_be_silently_replaced(override):
    with pytest.raises(ValueError):
        build_config(overrides=[override])


def test_manifest_allows_extended_budget_and_relocation_but_binds_model_and_assets(tmp_path):
    assets = tmp_path / 'assets'
    (assets / 'weights').mkdir(parents=True)
    (assets / 'weights/mineclip_attn.pth').write_bytes(b'mineclip fixture')
    unet = assets / 'affordance_map/finetune_unet/finetune_checkpoints/harvest_log_in_plains/swin_unet_checkpoint.pth'
    unet.parent.mkdir(parents=True)
    unet.write_bytes(b'unet fixture')
    first = build_config(stage='smoke', assets_root=assets, logdir=tmp_path / 'run')
    extended = build_config(stage='smoke', assets_root=assets, logdir=tmp_path / 'other',
                            overrides=['train_steps=10000'])
    identity = experiment_manifest(first)['identity']
    assert experiment_manifest(extended)['identity'] == identity
    changed = build_config(stage='smoke', assets_root=assets, logdir=tmp_path / 'run',
                           overrides=['evidence.rc_weight=0.0'])
    assert experiment_manifest(changed)['identity'] != identity
    unet.write_bytes(b'different unet')
    assert experiment_manifest(first)['identity'] != identity


def test_public_dry_run_selects_gpu_before_backend_and_preserves_clean_mode(tmp_path):
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'run_evidence.py'),
                             '--gpu', '1', '--seed', '1', '--logdir', str(tmp_path / 'run'),
                             '--checkpoint-every', '20000', '--dry-run'],
                            capture_output=True, text=True, check=True, env=os.environ.copy())
    plan = json.loads(result.stdout)
    assert plan['gpu'] == '1' and plan['config']['seed'] == 1
    assert plan['config']['device'] == plan['config']['env_device'] == 'cuda:0'
    assert plan['config']['protocol']['eval_action'] == 'mode'
    assert plan['config']['checkpoint_every'] == 20000
    assert plan['config']['logdir'] == str(tmp_path / 'run')


def test_training_rejects_checkpoint_step_instead_of_ignoring_it():
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'run_evidence.py'),
                             '--checkpoint-step', '20000', '--resume', '--dry-run'],
                            capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode != 0
    assert '--checkpoint-step is only supported with --eval-only' in result.stderr
