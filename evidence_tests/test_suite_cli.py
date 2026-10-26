"""Public command dispatch must preserve method, seed, GPU and output isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from run_ablation_suite import VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    # Exercise the executable entry from outside its source directory.
    return subprocess.run([sys.executable, str(ROOT / 'run_experiment.py'), *args],
                          cwd=ROOT.parent, capture_output=True, text=True, timeout=30)


def test_unified_cli_selects_method_gpu_and_noncolliding_output_directories():
    paths = set()
    for method, variants in [('evidence_dreamer', VARIANTS), ('ls_imagine', (None,))]:
        for variant in variants:
            args = ['--method', method, '--gpu', '1', '--seed', '3', '--dry-run']
            if variant:
                args += ['--ablation', variant]
            result = invoke(*args)
            assert result.returncode == 0, result.stderr
            payload = json.loads(result.stdout)
            config = payload['config']
            assert payload['gpu'] == '1'
            assert config['method'] == method and config['seed'] == 3
            assert config['device'] == config['env_device'] == 'cuda:0'
            assert config['logdir'] not in paths
            assert '/formal/' in config['logdir']
            paths.add(config['logdir'])
    assert len(paths) == len(VARIANTS) + 1


def test_ls_rejects_evidence_ablation_instead_of_silently_ignoring_it():
    result = invoke('--method', 'ls_imagine', '--ablation', 'no_rc', '--dry-run')
    assert result.returncode != 0
    assert 'unrecognized arguments' in result.stderr


def test_plan_is_read_only_and_uses_selected_interpreter():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'run_ablation_suite.py'), '--gpu', '1',
         '--seeds', '0', '1', '--ablations', 'full', 'no_rc', '--include-ls'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = result.stdout.strip().splitlines()
    assert len(rows) == 6
    assert sum('--method ls_imagine' in row for row in rows) == 2
    assert all('--gpu 1' in row for row in rows)
    assert all(sys.executable in row for row in rows)


def test_plan_rejects_duplicate_directory_targets():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'run_ablation_suite.py'), '--seeds', '1', '1'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Duplicate seeds or ablations' in result.stderr
