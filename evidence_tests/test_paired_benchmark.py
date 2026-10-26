"""Paired plans share budgets and seeds without reusing historical directories."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import run_paired_benchmark as paired


def option(command, name):
    return command[command.index(name) + 1]


def test_default_pair_plan_has_matching_seeds_budgets_and_separate_paths(tmp_path):
    args = paired.parse_args(['--experiment-root', str(tmp_path / 'paired'), '--gpu', '1'])
    runs = list(paired.planned_runs(args))
    assert len(runs) == 6 and len({p for p, _ in runs}) == 6
    assert not (tmp_path / 'paired').exists()
    for i, seed in enumerate((0, 1, 2)):
        ls, evidence = runs[2*i:2*i+2]
        assert option(ls[1], '--method') == 'ls_imagine'
        assert option(evidence[1], '--ablation') == 'no_memory'
        for _, command in (ls, evidence):
            assert command[0] == sys.executable
            assert option(command, '--seed') == str(seed)
            assert option(command, '--gpu') == '1'
            assert option(command, '--steps') == '1000000'
            assert option(command, '--eval-every') == '20000'
            assert option(command, '--eval-episodes') == '10'
            assert option(command, '--checkpoint-every') == '20000'


def test_diagnostic_switches_go_to_evidence_only_and_methods_can_use_separate_gpus(tmp_path):
    args = paired.parse_args(['--experiment-root', str(tmp_path), '--seeds', '2',
                             '--diagnostics-every', '40000', '--diagnostics-batch-size', '1',
                             '--no-diagnostic-gradients'])
    runs = list(paired.planned_runs(args))
    ls, evidence = [c for _, c in runs]
    assert '--set' not in ls
    assert 'diagnostics.every=40000' in evidence
    assert 'diagnostics.batch_size=1' in evidence
    assert 'diagnostics.gradients=false' in evidence
    filtered = paired.parse_args(['--experiment-root', str(tmp_path), '--seeds', '2',
                                  '--methods', 'no_memory', '--gpu', '1', '--no-diagnostics'])
    one = list(paired.planned_runs(filtered))
    assert len(one) == 1 and one[0][0] == runs[1][0]
    assert 'diagnostics.enabled=false' in one[0][1]
    assert option(one[0][1], '--gpu') == '1'


def test_generated_commands_use_actual_framework_configs_with_matched_protocol(tmp_path):
    args = paired.parse_args(['--experiment-root', str(tmp_path / 'fresh'), '--seeds', '3',
                             '--stage', 'smoke', '--gpu', '1', '--steps', '6000',
                             '--eval-every', '1000', '--eval-episodes', '2',
                             '--checkpoint-every', '1000'])
    configs = []
    for path, command in paired.planned_runs(args):
        result = subprocess.run(command + ['--dry-run'], cwd=tmp_path, capture_output=True,
                                text=True, timeout=30, check=True)
        plan = json.loads(result.stdout)
        c = plan['config']
        assert plan['gpu'] == '1' and c['seed'] == 3
        assert c['logdir'] == str(path)
        assert c['device'] == c['env_device'] == 'cuda:0'
        configs.append(c)
    ls, evidence = configs
    for ls_key, evidence_key in [('steps', 'train_steps'), ('eval_every', 'eval_every'),
                                 ('eval_episode_num', 'eval_episodes'),
                                 ('checkpoint_every', 'checkpoint_every'), ('prefill', 'prefill'),
                                 ('batch_size', 'batch_size'), ('batch_length', 'batch_length'),
                                 ('train_ratio', 'train_ratio')]:
        assert ls['clean'][ls_key] == evidence[evidence_key]
    assert not (tmp_path / 'fresh').exists()


def test_new_combination_controls_can_be_planned_in_same_root(tmp_path):
    args = paired.parse_args(['--experiment-root', str(tmp_path), '--seeds', '1',
                             '--methods', 'no_memory', 'no_memory_no_rc', 'no_memory_no_fc_rc'])
    runs = list(paired.planned_runs(args))
    assert [option(c, '--ablation') for _, c in runs] == args.methods
    assert len({p for p, _ in runs}) == 3


@pytest.mark.parametrize('extra', [['--seeds', '1', '1'],
                                 ['--methods', 'no_memory', 'no_memory'],
                                 ['--seeds', '-1'], ['--gpu', '0,1'],
                                 ['--diagnostics-every', '0'],
                                 ['--task', '../escape'], ['--execute', '--check']])
def test_invalid_plans_are_rejected(extra):
    with pytest.raises(SystemExit):
        paired.parse_args(extra)


def test_execute_rejects_later_nonempty_directory_before_starting_first_run(tmp_path, monkeypatch):
    args = ['--experiment-root', str(tmp_path), '--seeds', '1', '--execute']
    runs = list(paired.planned_runs(paired.parse_args(args)))
    runs[1][0].mkdir(parents=True)
    (runs[1][0] / 'metrics.jsonl').write_text('{}\n')
    calls = []
    monkeypatch.setattr(paired.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(FileExistsError):
        paired.main(args)
    assert calls == [] and not runs[0][0].exists()


def make_checkpoint_skeleton(path):
    checkpoint = path / 'checkpoints' / 'step_0000020000'
    checkpoint.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'provenance.json').write_text('{}')
    (path / 'checkpoints/latest.json').write_text(json.dumps({'directory': checkpoint.name}))
    (checkpoint / 'manifest.json').write_text('{}')
    # The suite checks structure only; the runner still verifies SHA/source identity.
    (checkpoint / 'state.pkl.gz').write_bytes(b'placeholder')


def test_resume_never_silently_starts_missing_run(tmp_path, monkeypatch):
    args = ['--experiment-root', str(tmp_path), '--seeds', '1', '--resume', '--execute']
    runs = list(paired.planned_runs(paired.parse_args(args)))
    make_checkpoint_skeleton(runs[0][0])
    calls = []
    monkeypatch.setattr(paired.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(FileNotFoundError):
        paired.main(args)
    assert calls == []
    make_checkpoint_skeleton(runs[1][0])
    paired.main(args)
    assert len(calls) == 2 and all('--resume' in call[0] for call in calls)


def test_resume_rejects_unsafe_checkpoint_pointer(tmp_path):
    make_checkpoint_skeleton(tmp_path)
    (tmp_path / 'checkpoints/latest.json').write_text('{"directory":"../other"}')
    with pytest.raises(ValueError, match='Invalid checkpoint pointer'):
        paired.validate_destinations([(tmp_path, [])], resume=True)


def test_execution_stops_on_first_failed_method(tmp_path, monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(paired.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        paired.main(['--experiment-root', str(tmp_path), '--seeds', '1', '--execute'])
    assert len(calls) == 1
