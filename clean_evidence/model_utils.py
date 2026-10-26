"""Model-only config utilities; the environment is owned by Clean LS."""
import copy
from pathlib import Path
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as file:
        value = YAML(typ='safe').load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f'Expected a mapping in {path}')
    return value

def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
