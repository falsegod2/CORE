#!/usr/bin/env python3
"""One serial entry point for Clean LS-Imagine and Evidence ablations."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument('--method', choices=('evidence_dreamer', 'ls_imagine'),
                        default='evidence_dreamer')
    args, remaining = parser.parse_known_args(argv)
    # These modules parse options and select CUDA visibility before importing
    # torch or MineDojo. Do not import their runners here.
    if args.method == 'ls_imagine':
        from run_ls_imagine import main as run
    else:
        from run_evidence import main as run
    return run(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
