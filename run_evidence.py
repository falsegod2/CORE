#!/usr/bin/env python3
"""EvidenceDreamer using the uploaded Clean LS environment and reset wrappers."""
import argparse
import json
import os
from pathlib import Path
import re

from clean_evidence.configuration import PROJECT_ROOT, build_config
from clean_evidence.ablations import ABLATIONS, descriptions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', default='minedojo_harvest_log_in_plains')
    parser.add_argument('--stage', choices=['formal', 'smoke'], default='formal')
    parser.add_argument('--ablation', choices=list(ABLATIONS), default='full')
    parser.add_argument('--list-ablations', action='store_true', help='Print named Evidence controls and exit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES') or '0')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--logdir', help='Exact run directory; reused with --resume')
    parser.add_argument('--assets-root')
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--set', action='append', default=[], metavar='PATH=VALUE')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate saved run from --logdir without training')
    parser.add_argument('--checkpoint-step', type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_ablations:
        print(json.dumps(descriptions(), ensure_ascii=False, indent=2))
        return 0
    if args.checkpoint_step is not None and not args.eval_only:
        raise ValueError('--checkpoint-step is only supported with --eval-only; --resume loads the latest complete checkpoint')
    if not re.fullmatch(r'(?:\d+|GPU-[\w-]+|MIG-[\w/-]+)', args.gpu):
        raise ValueError('--gpu must select one physical GPU')
    # Set visibility before any learning/teacher backend is imported.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MINEDOJO_HEADLESS'] = '1'
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.chdir(PROJECT_ROOT)
    if args.eval_only:
        if not args.logdir or args.resume or args.set:
            raise ValueError('--eval-only requires --logdir and does not accept --resume or --set')
        root = Path(args.logdir).expanduser().resolve()
        config = json.loads((root / 'config.json').read_text())
        config['logdir'] = str(root)
        config['device'] = config['env_device'] = 'cuda:0'
        config['clean']['device'] = 'cuda:0'
        if args.assets_root:
            config['assets_root'] = str(Path(args.assets_root).expanduser().resolve())
    else:
        overrides = []
        for name, key in [('steps', 'train_steps'), ('checkpoint_every', 'checkpoint_every'),
                          ('eval_every', 'eval_every'), ('eval_episodes', 'eval_episodes')]:
            value = getattr(args, name)
            if value is not None:
                overrides.append(f'{key}={value}')
        config = build_config(task=args.task, stage=args.stage, seed=args.seed,
                              logdir=args.logdir, assets_root=args.assets_root,
                              overrides=overrides + args.set, ablation=args.ablation)
    if config['device'] != 'cuda:0' or config['env_device'] != 'cuda:0':
        raise ValueError('Use --gpu for physical selection; the real Clean CUDA runtime uses logical cuda:0')
    if args.dry_run:
        print(json.dumps({'gpu': args.gpu, 'config': config}, ensure_ascii=False, indent=2))
        return 0
    from clean_evidence.runtime.preflight import check
    check(config)
    if args.check:
        return 0
    if args.eval_only:
        from clean_evidence.configuration import experiment_manifest
        from clean_evidence.storage import Checkpoints, RunLock
        from clean_evidence.metrics import Metrics
        from clean_evidence.methods.evidence_dreamer import EvidenceDreamer
        from clean_evidence.environment import CleanEvidenceEnv
        from clean_evidence.runner import evaluate
        with RunLock(root / '.run.lock'):
            meta = experiment_manifest(config)
            stored = json.loads((root / 'provenance.json').read_text())
            if stored['identity'] != meta['identity']:
                raise ValueError('Evaluation source/config/resource mismatch')
            state = Checkpoints(root / 'checkpoints').load(meta['identity'], args.checkpoint_step)
            method = EvidenceDreamer(config, root)
            env = None
            try:
                method.load_state_dict(state['method'])
                env = CleanEvidenceEnv(config, mode='eval', run_dir=root)
                evaluate(method, env, config, Metrics(root / 'metrics.jsonl'), state['step'],
                         episodes=args.eval_episodes)
            finally:
                if env is not None:
                    env.close()
                method.close()
        return 0
    from clean_evidence.runner import Trainer
    Trainer(config, resume=args.resume).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
