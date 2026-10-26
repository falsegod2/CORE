#!/usr/bin/env python3
"""Run original Clean LS-Imagine with managed serial scheduling and checkpoints."""
import argparse
import json
import os
import re
from clean_ls.configuration import PROJECT_ROOT, build_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', default='minedojo_harvest_log_in_plains')
    parser.add_argument('--stage', choices=['formal', 'smoke'], default='formal')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES') or '0')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--logdir')
    parser.add_argument('--assets-root')
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--set', action='append', default=[], metavar='PATH=VALUE')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not re.fullmatch(r'(?:\d+|GPU-[\w-]+|MIG-[\w/-]+)', args.gpu):
        raise ValueError('--gpu must select one physical GPU')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MINEDOJO_HEADLESS'] = '1'
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.chdir(PROJECT_ROOT)
    overrides = []
    for name, key in [('steps', 'steps'), ('checkpoint_every', 'checkpoint_every'),
                      ('eval_every', 'eval_every'), ('eval_episodes', 'eval_episode_num')]:
        value = getattr(args, name)
        if value is not None:
            overrides.append(f'clean.{key}={value}')
    config = build_config(task=args.task, seed=args.seed, stage=args.stage, logdir=args.logdir,
                          assets_root=args.assets_root, overrides=overrides + args.set)
    if config['device'] != 'cuda:0' or config['env_device'] != 'cuda:0':
        raise ValueError('Use --gpu for physical selection; logical device is cuda:0')
    if args.dry_run:
        print(json.dumps({'gpu': args.gpu, 'config': config}, ensure_ascii=False, indent=2))
        return 0
    from clean_evidence.runtime.preflight import check
    check(config)
    if config['clean']['use_wandb']:
        import wandb  # Optional logging dependency, only when explicitly enabled.
    if args.check:
        return 0
    from clean_ls.runner import Trainer
    Trainer(config, resume=args.resume).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
