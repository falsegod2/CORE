"""Learner/replay random streams only: no environment seed plan or map bank."""
import contextlib
import hashlib
import random
import numpy as np


def seed_for(root, *labels):
    token = '/'.join(map(str, (root, *labels))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little') & 2147483647

@contextlib.contextmanager
def isolated_rng(seed=None):
    """Save learner RNGs; scope this around policy work, not Clean environment resets."""
    import sys
    py = random.getstate()
    npstate = np.random.get_state()
    torch = sys.modules.get('torch')
    cpu = torch.get_rng_state() if torch else None
    cuda = torch.cuda.get_rng_state_all() if torch and torch.cuda.is_available() else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if torch:
            torch.manual_seed(seed)
            if cuda is not None:
                torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        if torch:
            torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
