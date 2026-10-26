"""Use existing local dependencies/assets; never install or rewrite a runtime."""
import glob
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ..configuration import PROJECT_ROOT, resource_paths


def java8_environment():
    candidates = []
    if os.environ.get('JAVA_HOME'):
        candidates.append(Path(os.environ['JAVA_HOME']).expanduser() / 'bin/java')
    current = shutil.which('java')
    if current:
        candidates.append(Path(current))
    cjj = Path.home() / 'CJJ'
    candidates += [cjj / 'miniconda3/envs/ls_imagine/bin/java', cjj / 'jdk8/bin/java']
    candidates += [Path(p) / 'bin/java' for p in sorted(glob.glob(str(cjj / 'jdk1.8*')))]
    candidates.append(Path('/usr/lib/jvm/java-8-openjdk-amd64/bin/java'))
    for path in candidates:
        path = path.resolve()
        if not path.is_file():
            continue
        try:
            result = subprocess.run([str(path), '-version'], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and re.search(r'version\s+"(?:1\.8(?:\.|\")|8(?:\.|\"))', result.stdout + result.stderr):
            home = path.parent.parent
            if home.name == 'jre' and (home.parent / 'bin/javac').is_file():
                home = home.parent
            os.environ['JAVA_HOME'] = str(home)
            os.environ['PATH'] = str(home / 'bin') + os.pathsep + os.environ.get('PATH', '')
            return str(home)
    raise RuntimeError('No existing Java 8 found. Set JAVA_HOME to your working JDK8 directory.')


def check(config):
    resources = resource_paths(config)
    for name, path in resources.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}: {path}; pass --assets-root to your existing CORE resource directory')
    java = java8_environment()
    names = ['numpy', 'torch', 'gym', 'minedojo', 'mineclip', 'omegaconf',
             'cv2', 'einops', 'timm', 'scipy', 'matplotlib', 'torchvision', 'tensorboard']
    modules = {}
    for name in names:
        try:
            modules[name] = importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(f'Cannot import {name} using {sys.executable}: {error}. Use the existing working ls_imagine environment; nothing was installed.') from error
    torch = modules['torch']
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Clean learner/MineCLIP runtime requires one visible CUDA device. Choose --gpu 0 or --gpu 1.')
    from .tokenizer import prepare
    tokenizer = prepare(PROJECT_ROOT, tokenizer=os.environ.get('MINECLIP_TOKENIZER'),
                        search_roots=[Path(config['assets_root'])])
    report = {'event': 'clean_runtime_ready', 'method': config.get('method', 'evidence_dreamer'),
              'ablation': config.get('ablation'), 'interpreter': sys.executable,
              'java_home': java, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'logical_device': config['device'], 'gpu': torch.cuda.get_device_name(0),
              'resources': {key: str(value) for key, value in resources.items()},
              'tokenizer': tokenizer,
              'environment_source': str(PROJECT_ROOT / 'envs/tasks/minedojo/__init__.py'),
              'not_validated': ['Minecraft startup/reset', 'GPU learner update', 'Learning benefit']}
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report
