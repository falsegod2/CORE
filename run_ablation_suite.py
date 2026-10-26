#!/usr/bin/env python3
"""Print a reproducible experiment plan; --execute runs it serially."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from clean_evidence.ablations import ABLATIONS

VARIANTS = tuple(ABLATIONS)
ROOT = Path(__file__).resolve().parent


def commands(args):
    common = ['--gpu', args.gpu, '--stage', args.stage, '--task', args.task]
    for field in ('steps', 'checkpoint_every', 'eval_every', 'eval_episodes'):
        value = getattr(args, field)
        if value is not None:
            common += ['--' + field.replace('_', '-'), str(value)]
    if args.assets_root:
        common += ['--assets-root', str(Path(args.assets_root).expanduser().resolve())]
    if args.resume:
        common += ['--resume']
    for seed in args.seeds:
        for variant in args.ablations:
            yield [sys.executable, '-u', str(ROOT / 'run_experiment.py'),
                   '--method', 'evidence_dreamer', '--ablation', variant,
                   '--seed', str(seed), *common]
        if args.include_ls:
            yield [sys.executable, '-u', str(ROOT / 'run_experiment.py'),
                   '--method', 'ls_imagine', '--seed', str(seed), *common]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--ablations', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--include-ls', action='store_true')
    parser.add_argument('--stage', choices=('formal', 'smoke'), default='formal')
    parser.add_argument('--task', default='minedojo_harvest_log_in_plains')
    parser.add_argument('--steps', type=int, help='Default: selected stage budget')
    parser.add_argument('--checkpoint-every', type=int, help='Formal default: 20000')
    parser.add_argument('--eval-every', type=int, help='Formal default: 20000')
    parser.add_argument('--eval-episodes', type=int, help='Formal default: 10')
    parser.add_argument('--assets-root')
    parser.add_argument('--resume', action='store_true',
                        help='All planned runs must already have valid checkpoints')
    parser.add_argument('--execute', action='store_true',
                        help='Actually run sequentially; stop on the first failed experiment')
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or len(args.ablations) != len(set(args.ablations)):
        parser.error('Duplicate seeds or ablations would reuse the same output directory')
    for command in commands(args):
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
