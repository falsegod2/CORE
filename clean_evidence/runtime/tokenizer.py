#!/usr/bin/env python3
"""Prepare MineCLIP's existing ./clip-vit-base-patch16 lookup using local files.

This helper never downloads files, patches an installed package, or replaces an
existing entry. Run it with the same interpreter and assets directory used by
the environment worker. A valid existing target always takes precedence.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


NAME = 'clip-vit-base-patch16'
TOKENIZER_FILES = ('vocab.json', 'merges.txt', 'tokenizer.json',
                   'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'config.json')


def fingerprint(path):
    """Hash the local tokenizer inputs, including configuration and token IDs."""
    path = Path(path)
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in TOKENIZER_FILES if (path / name).is_file()}


def local_candidates(workdir, extra=()):
    """Inspect a bounded set of existing CJJ locations and HF cache snapshots."""
    workdir = Path(workdir)
    cjj = Path.home() / 'CJJ'
    roots = [workdir, workdir.parent, cjj, cjj / 'LS-Imagine',
             cjj / 'DreamerV3_Clean_PixelOnly_BinaryReward' /
             'CORE-CORE_DREAMERV3_MINEDOJO', *map(Path, extra)]
    for key in ('CORE_ASSETS', 'WM_ASSET_ROOTS'):
        roots.extend(Path(p).expanduser() for p in
                     os.environ.get(key, '').split(os.pathsep) if p)
    try:
        spec = importlib.util.find_spec('mineclip')
    except (ImportError, ValueError):
        spec = None
    if spec and spec.origin:
        roots.extend(list(Path(spec.origin).resolve().parents)[:4])
    paths = []
    for root in roots:
        root = root.expanduser().resolve()
        paths.extend((root, root / NAME, root / 'MineCLIP' / NAME,
                      root / 'mineclip' / NAME, root / 'weights' / NAME,
                      root / 'MineCLIP' / 'mineclip' / NAME,
                      root / 'LS-Imagine-main' / NAME))
    caches = [Path.home() / '.cache' / 'huggingface' / 'hub']
    for key in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'):
        if os.environ.get(key):
            caches.append(Path(os.environ[key]).expanduser())
    if os.environ.get('HF_HOME'):
        caches.append(Path(os.environ['HF_HOME']).expanduser() / 'hub')
    for cache in caches:
        paths.extend(sorted((cache / 'models--openai--clip-vit-base-patch16' /
                             'snapshots').glob('*')))
    result = []
    seen = set()
    for path in paths:
        path = path.resolve()
        has_vocab = (path / 'vocab.json').is_file() and (path / 'merges.txt').is_file()
        if (path not in seen and path.is_dir() and
                (has_vocab or (path / 'tokenizer.json').is_file())):
            result.append(path)
            seen.add(path)
    return result


def validate_local(path):
    """Load only this absolute local directory and probe the original CLIP IDs."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(path).resolve()),
                                             local_files_only=True)
    if (tokenizer.vocab_size != 49408 or tokenizer.bos_token_id != 49406 or
            tokenizer.eos_token_id != 49407):
        raise RuntimeError('Expected CLIP vocabulary size 49408, BOS 49406, EOS 49407.')
    tokens = tokenizer(['Cut a tree', 'Collect wool'], padding='max_length',
                       truncation=True, max_length=77)['input_ids']
    if len(tokens) != 2 or any(len(row) != 77 for row in tokens):
        raise RuntimeError('Tokenizer failed the local 2 x 77 encoding check.')
    return {'class': type(tokenizer).__name__, 'vocab_size': tokenizer.vocab_size,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id, 'probe_shape': [2, 77],
            'local_files_only': True}


def prepare(workdir, tokenizer=None, search_roots=(), validator=validate_local,
            check_only=False):
    """Validate, then add one link only if MineCLIP's expected target is absent."""
    workdir = Path(workdir).expanduser().resolve()
    if not workdir.is_dir():
        raise RuntimeError(f'Working directory does not exist: {workdir}')
    explicit_value = tokenizer or os.environ.get('MINECLIP_TOKENIZER')
    explicit = Path(explicit_value).expanduser().resolve() if explicit_value else None
    if explicit is not None and not explicit.is_dir():
        raise RuntimeError(f'Explicit tokenizer directory does not exist: {explicit}')
    target = workdir / NAME
    if target.exists() or target.is_symlink():
        if not target.is_dir():
            raise RuntimeError(f'Refusing to replace existing entry: {target}')
        if explicit is not None and explicit != target.resolve() and fingerprint(explicit) != fingerprint(target):
            raise RuntimeError(f'Explicit tokenizer differs from existing target {target}. Nothing was replaced; select the existing tokenizer or use a separate assets directory.')
        probe = validator(target)
        return dict(status='already_ready', workdir=str(workdir),
                    tokenizer=str(target.resolve()), files=fingerprint(target),
                    validation=probe)
    candidates = [explicit] if explicit is not None else local_candidates(workdir, search_roots)
    if not candidates:
        raise RuntimeError('No complete local CLIP tokenizer found. Set '
                           'MINECLIP_TOKENIZER=/absolute/clip-vit-base-patch16 '
                           'before run_evidence.sh. No download was attempted.')
    fingerprints = [fingerprint(path) for path in candidates]
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        raise RuntimeError('Multiple different local tokenizers found; select one with '
                           'MINECLIP_TOKENIZER:\n' +
                           '\n'.join(map(str, candidates)))
    selected = candidates[0]
    probe = validator(selected)
    if check_only:
        return dict(status='validated_without_link', workdir=str(workdir),
                    tokenizer=str(selected), files=fingerprints[0], validation=probe)
    try:
        target.symlink_to(selected, target_is_directory=True)
    except FileExistsError:
        if not target.is_dir() or target.resolve() != selected:
            raise RuntimeError(f'Another process created a different entry at {target}; '
                               'nothing overwritten.')
    return dict(status='linked', workdir=str(workdir), tokenizer=str(selected),
                link=str(target), files=fingerprints[0], validation=probe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True,
                        help='Project root used as the Clean environment working directory.')
    parser.add_argument('--tokenizer', help='Existing local CLIP tokenizer directory.')
    parser.add_argument('--search-root', action='append', default=[])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    result = prepare(args.workdir, args.tokenizer, args.search_root,
                     check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'[TOKENIZER FAILED] {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
