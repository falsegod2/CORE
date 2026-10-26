#!/usr/bin/env python3
"""Plan or run matched-seed Clean LS / Evidence experiments serially.

All runs use run_experiment.py and explicit shared interaction/evaluation
budgets. Matching experiment seeds does not make Minecraft world sequences or
learned parameters identical. The original Clean reset protocol is preserved.
No training starts without --execute, and no existing run is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from clean_evidence.ablations import ABLATIONS


ROOT = Path(__file__).resolve().parent
METHODS = ('ls_imagine', *ABLATIONS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', default='0', help='One physical GPU for all selected runs')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        default=['ls_imagine', 'no_memory'],
                        help='Filter methods when running separate plans on different GPUs')
    parser.add_argument('--experiment-root', type=Path,
                        default=ROOT / 'logdir' / 'paired_20261025',
                        help='Dedicated root; each method/stage/task/seed has its own run directory')
    parser.add_argument('--stage', choices=('formal', 'smoke'), default='formal')
    parser.add_argument('--task', default='minedojo_harvest_log_in_plains')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--diagnostics-every', type=int,
                        help='Evidence only; default follows its stage configuration')
    parser.add_argument('--diagnostics-batch-size', type=int,
                        help='Evidence only; sequences per fixed diagnostic bank')
    parser.add_argument('--no-diagnostics', action='store_true',
                        help='Disable Evidence diagnostics explicitly, without changing losses')
    parser.add_argument('--no-diagnostic-gradients', action='store_true',
                        help='Keep Evidence forward diagnostics but skip gradient diagnostics')
    parser.add_argument('--resume', action='store_true',
                        help='Require existing checkpoints for every selected run; never silently restart')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true',
                        help='Run the printed commands sequentially, stopping at the first failure')
    action.add_argument('--check', action='store_true',
                        help='Run both frameworks\' preflight checks without training')
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or len(args.methods) != len(set(args.methods)):
        parser.error('Duplicate seeds or methods would reuse output directories')
    if any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be nonnegative')
    if not re.fullmatch(r'(?:\d+|GPU-[\w-]+|MIG-[\w/-]+)', args.gpu):
        parser.error('--gpu must select one physical GPU')
    if not re.fullmatch(r'minedojo_[A-Za-z0-9_]+', args.task):
        parser.error('--task must be a named minedojo task')
    defaults = dict(steps=1000000, eval_every=20000, eval_episodes=10, checkpoint_every=20000)
    if args.stage == 'smoke':
        defaults.update(steps=5000, eval_every=2500, eval_episodes=1, checkpoint_every=2500)
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    for name in (*defaults, 'diagnostics_every', 'diagnostics_batch_size'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    args.experiment_root = args.experiment_root.expanduser().resolve()
    if args.assets_root is not None:
        args.assets_root = args.assets_root.expanduser().resolve()
    return args


def planned_runs(args):
    common = ['--gpu', args.gpu, '--stage', args.stage, '--task', args.task]
    for name in ('steps', 'eval_every', 'eval_episodes', 'checkpoint_every'):
        common += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if args.assets_root is not None:
        common += ['--assets-root', str(args.assets_root)]
    if args.resume:
        common += ['--resume']
    if args.check:
        common += ['--check']
    for seed in args.seeds:
        for variant in args.methods:
            path = args.experiment_root / variant / args.stage / args.task / f'seed_{seed}'
            method = 'ls_imagine' if variant == 'ls_imagine' else 'evidence_dreamer'
            command = [sys.executable, '-u', str(ROOT / 'run_experiment.py'),
                       '--method', method, '--seed', str(seed), '--logdir', str(path), *common]
            if method == 'evidence_dreamer':
                command += ['--ablation', variant]
                for field, config_key in [('diagnostics_every', 'every'),
                                          ('diagnostics_batch_size', 'batch_size')]:
                    value = getattr(args, field)
                    if value is not None:
                        command += ['--set', f'diagnostics.{config_key}={value}']
                if args.no_diagnostics:
                    command += ['--set', 'diagnostics.enabled=false']
                if args.no_diagnostic_gradients:
                    command += ['--set', 'diagnostics.gradients=false']
            yield path, command


def validate_destinations(runs, resume):
    """Check all destinations before starting any run; runners enforce identity/locks.

    This does not deserialize checkpoints or replace the runner's checksum and
    source/config/resource identity checks. It catches mixed fresh/resume plans
    before an earlier run can consume hours of compute.
    """
    for path, _ in runs:
        if resume:
            for name in ('config.json', 'provenance.json', 'checkpoints/latest.json'):
                if not (path / name).is_file():
                    raise FileNotFoundError(f'Cannot resume {path}: missing {name}')
            pointer = json.loads((path / 'checkpoints/latest.json').read_text())
            name = pointer.get('directory', '')
            if not isinstance(name, str) or not re.fullmatch(r'step_\d{10}', name):
                raise ValueError(f'Invalid checkpoint pointer in {path}')
            checkpoint = path / 'checkpoints' / name
            if not (checkpoint / 'manifest.json').is_file() or not (checkpoint / 'state.pkl.gz').is_file():
                raise FileNotFoundError(f'Cannot resume {path}: checkpoint is incomplete')
        elif path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f'Run directory is not empty: {path}. Use --resume or another --experiment-root.')


def main(argv=None):
    args = parse_args(argv)
    runs = list(planned_runs(args))
    # Planning and --check are read-only with respect to run directories.
    if args.execute:
        validate_destinations(runs, args.resume)
    for _, command in runs:
        print(shlex.join(command), flush=True)
        if args.execute or args.check:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
